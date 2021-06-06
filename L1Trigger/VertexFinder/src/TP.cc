#include "L1Trigger/VertexFinder/interface/TP.h"

namespace l1tVertexFinder {

  TP::TP(const TrackingParticle* tpPtr, const AnalysisSettings& settings)
      : trackingParticle_(tpPtr), settings_(&settings) {
    const std::vector<SimTrack>& vst = tpPtr->g4Tracks();
    EncodedEventId eid = vst.at(0).eventId();
    inTimeBx_ = (eid.bunchCrossing() == 0);  // TP from in-time or out-of-time Bx.
    physicsCollision_ = (eid.event() == 0);  // TP from physics collision or from pileup.

    this->fillUse();        // Fill use_ flag, indicating if TP is worth keeping.
    this->fillUseForEff();  // Fill useForEff_ flag, indicating if TP is good for tracking efficiency measurement.
  }

  //=== Fill truth info with association from tracking particle to stubs.

  void TP::setMatchingStubs(const std::vector<const Stub*>& vMatchingStubs) {
    assocStubs_ = vMatchingStubs;

    this->fillUseForAlgEff();  // Fill useForAlgEff_ flag.
    this->fillUseForVertexReco();
    this->calcNumLayers();  // Calculate number of tracker layers this TP has stubs in.
  }

  void TP::fillUseForVertexReco() {
    useForVertexReco_ = false;
    if (use_) {
      useForVertexReco_ = settings_->tpsUseForVtxReco()(trackingParticle_);
    }

    if (useForVertexReco_) {
      useForVertexReco_ = (utility::countLayers(*settings_, assocStubs_) >= settings_->genMinStubLayers() and
                           utility::countLayers(*settings_, assocStubs_, true) >= 2);
    }
  }

  //=== Check if this tracking particle is worth keeping.
  void TP::fillUse() {
    // Use looser cuts here those those used for tracking efficiency measurement.
    // Keep only those TP that have a chance (allowing for finite track resolution) of being reconstructed as L1 tracks. L1 tracks not matching these TP will be defined as fake.
    use_ = settings_->tpsUse()(trackingParticle_);
  }

  //=== Check if this tracking particle can be used to measure the L1 tracking efficiency.
  void TP::fillUseForEff() {
    useForEff_ = false;
    if (use_) {
      useForEff_ = settings_->tpsUseForEff()(trackingParticle_);
    }
  }

  //=== Check if this tracking particle can be used to measure the L1 tracking algorithmic efficiency (makes stubs in enough layers).
  void TP::fillUseForAlgEff() {
    useForAlgEff_ = false;
    if (useForEff_) {
      useForAlgEff_ = (utility::countLayers(*settings_, assocStubs_) >= settings_->genMinStubLayers());
    }
  }

}  // namespace l1tVertexFinder
