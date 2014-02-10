#ifndef CommonTools_ParticleFlow_PFCandidateFwdPtrCollectionOverlapFilter_h
#define CommonTools_ParticleFlow_PFCandidateFwdPtrCollectionOverlapFilter_h


/**
  \class    PFCandidateFwdPtrCollectionOverlapFilter PFCandidateFwdPtrCollectionOverlapFilter.h "CommonTools/ParticleFlow/interface/PFCandidateFwdPtrCollectionOverlapFilter.h"
  \brief    Removes PFCandidates from collection "src" which overlap in dR and dPt with objects in "overlap" collection.
            Modeled after FwdPtrCollectionFilter in CommonTools/UtilAlgos

  \author   Dominick Olivito
*/


#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"

class PFCandidateFwdPtrCollectionOverlapFilter : public edm::EDFilter {
public :
  explicit PFCandidateFwdPtrCollectionOverlapFilter( edm::ParameterSet const & params );
  ~PFCandidateFwdPtrCollectionOverlapFilter() {}

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  virtual bool filter(edm::Event & iEvent, const edm::EventSetup& iSetup);
  bool isOverlap(const reco::PFCandidate& obj1, const reco::Candidate& obj2) const;

protected :
  edm::EDGetTokenT< std::vector< edm::FwdPtr<reco::PFCandidate> > > srcToken_;
  edm::EDGetTokenT< edm::View<reco::PFCandidate>  > srcViewToken_;
  edm::EDGetTokenT< edm::View<reco::Candidate>  > overlapToken_;
  bool          filter_;
  bool          makeClones_;
  double        maxDeltaR_;
  double        maxDPtRel_;
};

#endif
