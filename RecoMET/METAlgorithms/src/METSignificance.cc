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

}

metsig::METSignificance::~METSignificance() {
  delete ptRes_;
  delete phiRes_;
}


reco::METCovMatrix
metsig::METSignificance::getCovariance(const edm::View<reco::Jet>& jets,
				       const std::vector< edm::Handle<reco::CandidateView> >& leptons,
				       const edm::View<reco::Candidate>& pfCandidates) {
  
   // metsig covariance
   double cov_xx = 0;
   double cov_xy = 0;
   double cov_yy = 0;
 
   // for lepton and jet subtraction
   std::vector<reco::CandidatePtr> footprint;

   // subtract leptons out of sumPt
   for ( std::vector< edm::Handle<reco::CandidateView> >::const_iterator lep_i = leptons.begin();
         lep_i != leptons.end(); ++lep_i ) {
      for( reco::CandidateView::const_iterator lep = (*lep_i)->begin(); lep != (*lep_i)->end(); lep++ ){
         if( lep->pt() > 10 ){
            for( unsigned int n=0; n < lep->numberOfSourceCandidatePtrs(); n++ ){
               if( lep->sourceCandidatePtr(n).isNonnull() and lep->sourceCandidatePtr(n).isAvailable() ){
                  footprint.push_back(lep->sourceCandidatePtr(n));
               } 
            }
         }
      }
   }
   // subtract jets out of sumPt
   for(edm::View<reco::Jet>::const_iterator jet = jets.begin(); jet != jets.end(); ++jet) {

      // disambiguate jets and leptons
      if(!cleanJet(*jet, leptons) ) continue;

      for( unsigned int n=0; n < jet->numberOfSourceCandidatePtrs(); n++){
         if( jet->sourceCandidatePtr(n).isNonnull() and jet->sourceCandidatePtr(n).isAvailable() ){
            footprint.push_back(jet->sourceCandidatePtr(n));
         }
      }

   }

   // calculate sumPt
   double sumPt = 0;
   for( edm::View<reco::Candidate>::const_iterator cand = pfCandidates.begin();
         cand != pfCandidates.end(); ++cand){

      // check if candidate exists in a lepton or jet
      bool cleancand = true;
      for(unsigned int i=0; i < footprint.size(); i++){
         if( footprint[i]->p4() == cand->p4() ){
            cleancand = false;
         }
      }
      // if not, add to sumPt
      if( cleancand ){
         sumPt += cand->pt();
      }

   }

   // add jets to metsig covariance matrix and subtract them from sumPt
   for(edm::View<reco::Jet>::const_iterator jet = jets.begin(); jet != jets.end(); ++jet) {
     
     // disambiguate jets and leptons
     if(!cleanJet(*jet, leptons) ) continue;

      double jpt  = jet->pt();
      double jeta = jet->eta();
      double feta = std::abs(jeta);
      double c = jet->px()/jet->pt();
      double s = jet->py()/jet->pt();

      // jet energy resolutions
      double jeta_res = (std::abs(jeta) < 9.9) ? jeta : 9.89; // JetResolutions defined for |eta|<9.9
      double sigmapt = ptRes_->parameterEtaEval("sigma",jeta_res,jpt);
      double sigmaphi = phiRes_->parameterEtaEval("sigma",jeta_res,jpt);

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

      } else {

         // add the (corrected) jet to the sumPt
         sumPt += jpt;

      }

   }


   //protection against unphysical events
   if(sumPt<0) sumPt=0;

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
      const std::vector< edm::Handle<reco::CandidateView> >& leptons ){

  for ( std::vector< edm::Handle<reco::CandidateView> >::const_iterator lep_i = leptons.begin();
        lep_i != leptons.end(); ++lep_i ) {
     for( reco::CandidateView::const_iterator lep = (*lep_i)->begin(); lep != (*lep_i)->end(); lep++ ){
        if ( reco::deltaR2(*lep, jet) < dR2match_ ) {
           return false;
        }
     }
  }
  return true;
}
