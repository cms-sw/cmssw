#ifndef RECOMET_METALGORITHMS_SIGNALGORESOLUTIONS_H
#define RECOMET_METALGORITHMS_SIGNALGORESOLUTIONS_H

// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SignAlgoResolutions
// 
/**\class METSignificance SignAlgoResolutions.h RecoMET/METAlgorithms/include/SignAlgoResolutions.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Kyle Story, Freya Blekman (Cornell University)
//         Created:  Fri Apr 18 11:58:33 CEST 2008
// $Id: SignAlgoResolutions.h,v 1.8 2013/05/06 17:56:33 sakuma Exp $
//
//

#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/SigInputObj.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "CondFormats/JetMETObjects/interface/JetResolution.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

#include <map>
#include <iostream>
#include <vector>

namespace metsig {

  enum resolutionType { caloEE, caloEB, caloHE, caloHO, caloHF, caloHB, jet, electron, tau, muon,PFtype1,PFtype2, PFtype3, PFtype4, PFtype5, PFtype6, PFtype7 };
  enum resolutionFunc { ET, PHI,TRACKP,CONSTPHI };

  class SignAlgoResolutions{
    
  public:
    SignAlgoResolutions():functionmap_(){;}
    SignAlgoResolutions(const edm::ParameterSet &iConfig);

    void addResolutions(const edm::ParameterSet &iConfig);
    double eval(const resolutionType & type, const resolutionFunc & func, const double & et, const double & phi, const double & eta, const double &p) const; // for example getvalue(caloHF,ET,et,phi,eta,p);
    double eval(const resolutionType & type, const resolutionFunc & func, const double & et, const double & phi, const double & eta) const; // for example getvalue(caloHF,ET,et,phi,eta,p);
    metsig::SigInputObj evalPF(const reco::PFCandidate* candidate) const;
    metsig::SigInputObj evalPFJet(const reco::PFJet *jet) const;
    bool isFilled() const {return functionmap_.size()>0;}
    
  private:
    double getfunc(const resolutionType & type,const resolutionFunc & func,  std::vector<double> & x) const;
    void addfunction(const resolutionType type, const resolutionFunc func, std::vector<double> parameters);
    void initializeJetResolutions( const edm::ParameterSet &iConfig );
    
    typedef std::pair<metsig::resolutionType, metsig::resolutionFunc> functionCombo;
    typedef std::vector<double> functionPars;
    std::map<functionCombo,functionPars> functionmap_;
 
    double EtFunction( const functionPars &x,  const functionPars &  par) const;
    double PhiFunction( const functionPars &x,  const functionPars & par) const;
    double PFunction( const functionPars &x, const functionPars &par) const;
    double PhiConstFunction(const functionPars &x, const functionPars &par) const;
    double ElectronPtResolution(const reco::PFCandidate *c) const;

    double ptResolThreshold_;
    //temporary fix for low pT jet resolutions
    //First index, eta bins, from 0 to 5;
    //Second index, pt bins, from 3 to 23 GeV;
    std::vector<double> jdpt[10];
    std::vector<double> jdphi[10];
     
    JetResolution *ptResol_;
    JetResolution *phiResol_;
    PFEnergyResolution *pfresol_;
  };
}
#endif
