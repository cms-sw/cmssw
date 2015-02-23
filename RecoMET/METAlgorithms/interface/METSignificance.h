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
#ifndef METAlgorithms_METSignificance_h
#define METAlgorithms_METSignificance_h
//____________________________________________________________________________||
#include "CondFormats/JetMETObjects/interface/JetResolution.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TF1.h"

//____________________________________________________________________________||
namespace metsig {

   class METSignificance {
      public:
         METSignificance(const edm::ParameterSet& iConfig);
         ~METSignificance();

         reco::METCovMatrix getCovariance(const edm::View<reco::Jet>& jets,
					  const std::vector< edm::Handle<reco::CandidateView> >& leptons,
					  const edm::View<reco::Candidate>& pfCandidates);
     double getSignificance(const reco::METCovMatrix& cov, const reco::MET& met ) const;

      private:
         bool cleanJet(const reco::Jet& jet, 
         const std::vector< edm::Handle<reco::CandidateView> >& leptons );

         double jetThreshold_;
         double dR2match_;
         std::vector<double> jetEtas_;
         std::vector<double> jetParams_;
         std::vector<double> pjetParams_;

         JetResolution* ptRes_;
         JetResolution* phiRes_;

   };

}

//____________________________________________________________________________||
#endif // METAlgorithms_METSignificance_h
