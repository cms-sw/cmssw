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

#include "RecoMET/METAlgorithms/interface/METSignificance.h"


metsig::METSignificance::METSignificance(const edm::ParameterSet& iConfig) {

  edm::ParameterSet cfgParams = iConfig.getParameter<edm::ParameterSet>("parameters");


  std::string ptResFileName = cfgParams.getParameter<std::string>("ptResFile");
  std::string phiResFileName = cfgParams.getParameter<std::string>("phiResFile");

  double dRmatch = cfgParams.getParameter<double>("dRMatch");
  dR2match_ = dRmatch*dRmatch;
  
  jetThreshold_ = cfgParams.getParameter<double>("jetThreshold");
  jetEtas_ = cfgParams.getParameter<std::vector<double> >("jeta");
  jetParams_ = cfgParams.getParameter<std::vector<double> >("jpar");
  pjetParams_ = cfgParams.getParameter<std::vector<double> >("pjpar");


  edm::FileInPath fpt("CondFormats/JetMETObjects/data/"+ptResFileName);
  edm::FileInPath fphi("CondFormats/JetMETObjects/data/"+phiResFileName);

  ptRes_  = new JetResolution(fpt.fullPath().c_str(),false);
  phiRes_ = new JetResolution(fphi.fullPath().c_str(),false);

  fPtEta_ = nullptr;
  fPhiEta_ = nullptr;

}

metsig::METSignificance::~METSignificance() {
  delete ptRes_;
  delete phiRes_;
  delete fPtEta_;
  delete fPhiEta_;
}


reco::METCovMatrix
metsig::METSignificance::getCovariance(const edm::View<reco::Jet>& jets,
				       const std::vector<reco::Candidate::LorentzVector>& leptons,
				       const edm::View<reco::Candidate>& pfCandidates) {
  
   // metsig covariance
   double cov_xx = 0;
   double cov_xy = 0;
   double cov_yy = 0;
 
   // calculate sumPt
   double sumPt = 0;
   for( edm::View<reco::Candidate>::const_iterator cand = pfCandidates.begin();
         cand != pfCandidates.end(); ++cand){
      sumPt += cand->pt();
   }

   //MM & NM :subtraction of lepton will need more study to see which option is optimal
   //does not signficantly change the results

   // subtract leptons out of sumPt
   // for ( std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin();
   //       lepton != leptons.end(); ++lepton ) {
   //   //sumPt -= lepton->Pt();
   // }
   // //protection against unphysical events
   // if(sumPt<0) sumPt=0;

   // add jets to metsig covariance matrix and subtract them from sumPt
   for(edm::View<reco::Jet>::const_iterator jet = jets.begin(); jet != jets.end(); ++jet) {
     
     // disambiguate jets and leptons
     if(!cleanJet(*jet, leptons) ) continue;

      double jpt  = jet->pt();
      double jeta = jet->eta();
      double feta = std::abs(jeta);
      double c = std::cos(jet->phi());
      double s = std::sin(jet->phi());

      // jet energy resolutions
      double jeta_res = (std::abs(jeta) < 9.9) ? jeta : 9.89; // JetResolutions defined for |eta|<9.9
      fPtEta_    = ptRes_ -> parameterEta("sigma",jeta_res);
      fPhiEta_   = phiRes_-> parameterEta("sigma",jeta_res);
      double sigmapt = fPtEta_->Eval(jpt);
      double sigmaphi = fPhiEta_->Eval(jpt);
      delete fPtEta_;
      delete fPhiEta_;

      // split into high-pt and low-pt sector
      if( jpt > jetThreshold_ ){
         // high-pt jets enter into the covariance matrix via JER

         double scale = 0;
         if(feta<jetEtas_[0]) scale = jetParams_[0];
         else if(feta<jetEtas_[1]) scale = jetParams_[1];
         else if(feta<jetEtas_[2]) scale = jetParams_[2];
         else if(feta<jetEtas_[3]) scale = jetParams_[3];
         else scale = jetParams_[4];

         double dpt = scale*jpt*sigmapt;
         double dph = jpt*sigmaphi;

         cov_xx += dpt*dpt*c*c + dph*dph*s*s;
         cov_xy += (dpt*dpt-dph*dph)*c*s;
         cov_yy += dph*dph*c*c + dpt*dpt*s*s;

         // subtract the pf constituents in each jet out of the sumPt
         for(unsigned int i=0; i < jet->numberOfDaughters(); i++){
            sumPt -= jet->daughter(i)->pt();
         }

      } else {

         // subtract the pf constituents in each jet out of the sumPt
         for(unsigned int i=0; i < jet->numberOfDaughters(); i++){
            sumPt -= jet->daughter(i)->pt();
         }
         // add the (corrected) jet to the sumPt
         sumPt += jpt;

      }

   }

   // add pseudo-jet to metsig covariance matrix
   cov_xx += pjetParams_[0]*pjetParams_[0] + pjetParams_[1]*pjetParams_[1]*sumPt;
   cov_yy += pjetParams_[0]*pjetParams_[0] + pjetParams_[1]*pjetParams_[1]*sumPt;

   reco::METCovMatrix cov;
   cov(0,0) = cov_xx;
   cov(1,0) = cov_xy;
   cov(0,1) = cov_xy;
   cov(1,1) = cov_yy;

   return cov;
}

double
metsig::METSignificance::getSignificance(const reco::METCovMatrix& cov, const reco::MET& met) const {

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

bool
metsig::METSignificance::cleanJet(const reco::Jet& jet, 
				  const std::vector<reco::Candidate::LorentzVector>& leptons) {
  
  if ( jet.pt() < jetThreshold_ ) return false;

  for ( std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin();
	lepton != leptons.end(); ++lepton ) {
    if ( reco::deltaR2( *lepton, jet) < dR2match_ ) {
      return false;
    }
  }
  return true;
}
