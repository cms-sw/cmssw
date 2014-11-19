// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      METSignificance
// 
/**\class METSignificance METSignificance.cc RecoMET/METAlgorithms/src/METSignificance.cc
Description: [one line class summary]
Implementation:
[Notes on implementation]
*/
//
// Original Author:  Nathan Mirman (Cornell University)
//         Created:  Thu May 30 16:39:52 CDT 2013
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/JetMETObjects/interface/JetResolution.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "TLorentzVector.h"
#include "TFile.h"
#include "TTree.h"
#include "TMatrixD.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "RecoMET/METAlgorithms/interface/METSignificance.h"

TMatrixD
metsig::METSignificance::getCovariance()
{

   edm::FileInPath fpt(ptResFileName);
   edm::FileInPath fphi(phiResFileName);

   JetResolution *ptRes_  = new JetResolution(fpt.fullPath().c_str(),false);
   JetResolution *phiRes_ = new JetResolution(fphi.fullPath().c_str(),false);

   // disambiguate jets and leptons
   std::vector<reco::Jet> cleanjets = cleanJets(jetThreshold, 0.4);

   // metsig covariance
   double cov_xx = 0;
   double cov_xy = 0;
   double cov_yy = 0;

   // calculate sumPt
   double sumPt = 0;
   for( std::vector<reco::Candidate::LorentzVector>::const_iterator cand = candidates.begin();
         cand != candidates.end(); ++cand){
      sumPt += cand->Pt();
   }

   // subtract leptons out of sumPt
   for ( std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin();
         lepton != leptons.end(); ++lepton ) {
      sumPt -= lepton->Pt();
   }

   // add jets to metsig covariance matrix and subtract them from sumPt
   for(std::vector<reco::Jet>::const_iterator jet = cleanjets.begin(); jet != cleanjets.end(); ++jet) {
      double jpt  = jet->pt();
      double jeta = jet->eta();
      double feta = fabs(jeta);
      double c = cos(jet->phi());
      double s = sin(jet->phi());

      // jet energy resolutions
      double jeta_res = (fabs(jeta) < 9.9) ? jeta : 9.89; // JetResolutions defined for |eta|<9.9
      TF1* fPtEta    = ptRes_ -> parameterEta("sigma",jeta_res);
      TF1* fPhiEta   = phiRes_-> parameterEta("sigma",jeta_res);
      double sigmapt = fPtEta->Eval(jpt);
      double sigmaphi = fPhiEta->Eval(jpt);
      delete fPtEta;
      delete fPhiEta;

      // split into high-pt and low-pt sector
      if( jpt > jetThreshold ){
         // high-pt jets enter into the covariance matrix via JER

         double scale = 0;
         if(feta<jetetas[0]) scale = jetparams[0];
         else if(feta<jetetas[1]) scale = jetparams[1];
         else if(feta<jetetas[2]) scale = jetparams[2];
         else if(feta<jetetas[3]) scale = jetparams[3];
         else scale = jetparams[4];

         double dpt = scale*jpt*sigmapt;
         double dph = jpt*sigmaphi;

         cov_xx += dpt*dpt*c*c + dph*dph*s*s;
         cov_xy += (dpt*dpt-dph*dph)*c*s;
         cov_yy += dph*dph*c*c + dpt*dpt*s*s;

         // subtract the pf constituents in each jet out of the sumPt
         for(unsigned int i=0; i < jet->numberOfDaughters(); i++){
            sumPt -= jet->daughter(i)->pt();
         }

      }else{

         // subtract the pf constituents in each jet out of the sumPt
         for(unsigned int i=0; i < jet->numberOfDaughters(); i++){
            sumPt -= jet->daughter(i)->pt();
         }
         // add the (corrected) jet to the sumPt
         sumPt += jpt;

      }

   }

   // add pseudo-jet to metsig covariance matrix
   cov_xx += pjetparams[0]*pjetparams[0] + pjetparams[1]*pjetparams[1]*sumPt;
   cov_yy += pjetparams[0]*pjetparams[0] + pjetparams[1]*pjetparams[1]*sumPt;

   TMatrixD cov(2,2);
   cov(0,0) = cov_xx;
   cov(1,0) = cov_xy;
   cov(0,1) = cov_xy;
   cov(1,1) = cov_yy;

   return cov;
}

double
metsig::METSignificance::getSignificance(TMatrixD& cov)
{
   // covariance matrix determinant
   double det = cov(0,0)*cov(1,1) - cov(0,1)*cov(1,0);

   // invert matrix
   double ncov_xx = cov(1,1) / det;
   double ncov_xy = -cov(0,1) / det;
   double ncov_yy = cov(0,0) / det;

   // product of met and inverse of covariance
   double sig = met.px()*met.px()*ncov_xx + 2*met.px()*met.py()*ncov_xy + met.py()*met.py()*ncov_yy;

   return sig;
}

std::vector<reco::Jet>
metsig::METSignificance::cleanJets(double ptThreshold, double dRmatch)
{
   double dR2match = dRmatch*dRmatch;
   std::vector<reco::Jet> retVal;
   for ( std::vector<reco::Jet>::const_iterator jet = jets.begin();
         jet != jets.end(); ++jet ) {
      bool isOverlap = false;
      for ( std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin();
            lepton != leptons.end(); ++lepton ) {
         TLorentzVector ljet, llep;
         ljet.SetPtEtaPhiE( jet->pt(), jet->eta(), jet->phi(), jet->energy() );
         llep.SetPtEtaPhiE( lepton->pt(), lepton->eta(), lepton->phi(), lepton->energy() );
         if ( pow(ljet.DeltaR( llep ),2) < dR2match ) isOverlap = true;  
      }
      if ( jet->pt() > ptThreshold && !isOverlap ){
         retVal.push_back(*jet);
      }
   }

   return retVal;
}

void
metsig::METSignificance::addJets(const std::vector<reco::Jet>& inputJets){
   jets = inputJets;
}

void
metsig::METSignificance::addLeptons(const std::vector<reco::Candidate::LorentzVector>& inputLeptons){
   leptons = inputLeptons;
}

void
metsig::METSignificance::addCandidates(const std::vector<reco::Candidate::LorentzVector>& inputCandidates){
   candidates = inputCandidates;
}

void
metsig::METSignificance::addMET(const reco::MET& inputMET){
   met = inputMET;
}

void
metsig::METSignificance::setThreshold(const double& inputThreshold){
   jetThreshold = inputThreshold;
}

void
metsig::METSignificance::setJetEtaBins(const std::vector<double>& inputEtaBins){
   jetetas = inputEtaBins;
}

void
metsig::METSignificance::setJetParams(const std::vector<double>& inputParams){
   jetparams = inputParams;
}

void
metsig::METSignificance::setPJetParams(const std::vector<double>& inputParams){
   pjetparams = inputParams;
}

void
metsig::METSignificance::setResFiles(const std::string& inputPtFile, const std::string& inputPhiFile){
   ptResFileName = inputPtFile;
   phiResFileName = inputPhiFile;
}

