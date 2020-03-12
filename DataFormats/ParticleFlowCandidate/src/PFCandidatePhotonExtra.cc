#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"
#include <ostream>
#include <iomanip>

using namespace reco;

PFCandidatePhotonExtra::PFCandidatePhotonExtra() {
  Mustache_Et_ = 0.;
  Excluded_clust_ = 0;

  GlobalCorr_ = 0.;
  GlobalCorrEError_ = 0.;
}

PFCandidatePhotonExtra::PFCandidatePhotonExtra(const reco::SuperClusterRef& scRef) {
  scRef_ = scRef;

  Mustache_Et_ = 0.;
  Excluded_clust_ = 0;

  GlobalCorr_ = 0.;
  GlobalCorrEError_ = 0.;
}

void PFCandidatePhotonExtra::addSingleLegConvTrackRef(const reco::TrackRef& trackref) {
  assoSingleLegRefTrack_.push_back(trackref);
}

void PFCandidatePhotonExtra::addSingleLegConvMva(float& mvasingleleg) { assoSingleLegMva_.push_back(mvasingleleg); }

void PFCandidatePhotonExtra::addConversionRef(const reco::ConversionRef& convref) {
  assoConversionsRef_.push_back(convref);
}

void PFCandidatePhotonExtra::addLCorrClusEnergy(float LCorrE) { LocalCorr_.push_back(LCorrE); }
