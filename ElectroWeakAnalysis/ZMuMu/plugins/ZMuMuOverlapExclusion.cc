#include "PhysicsTools/UtilAlgos/interface/OverlapExclusionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

struct ZMuMuOverlap {
  ZMuMuOverlap(const edm::ParameterSet&) { }
  bool operator()(const reco::Candidate & zMuMu, const reco::Candidate & z) const {
    return true;
  }
};

typedef SingleObjectSelector<
  edm::View<reco::Candidate>,
  OverlapExclusionSelector<reco::CandidateView, 
			   reco::Candidate, 
			   ZMuMuOverlap>
  > ZMuMuOverlapExclusionSelector;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMuOverlapExclusionSelector);


