#include "L1Trigger/VertexFinder/interface/TP.h"

namespace l1tVertexFinder {

  TP::TP()
      : settings_(nullptr),
        inTimeBx_(false),
        physicsCollision_(false),
        use_(false),
        useForEff_(false),
        useForAlgEff_(false),
        useForVertexReco_(false),
        nLayersWithStubs_(0) {}

  TP::TP(const TrackingParticlePtr& tpPtr, const AnalysisSettings& settings)
      : trackingParticle_(tpPtr), settings_(&settings) {
    const std::vector<SimTrack>& vst = trackingParticle_->g4Tracks();
    EncodedEventId eid = vst.at(0).eventId();
    inTimeBx_ = (eid.bunchCrossing() == 0);  // TP from in-time or out-of-time Bx.
    physicsCollision_ = (eid.event() == 0);  // TP from physics collision or from pileup.

    this->fillUse();        // Fill use_ flag, indicating if TP is worth keeping.
    this->fillUseForEff();  // Fill useForEff_ flag, indicating if TP is good for tracking efficiency measurement.
  }

  //=== Fill truth info with association from tracking particle to stubs.
  void TP::setMatchingStubs(const std::vector<Stub>& vMatchingStubs) {
    assocStubs_ = vMatchingStubs;

    this->fillUseForAlgEff();  // Fill useForAlgEff_ flag.
    this->fillUseForVertexReco();
    // Calculate number of tracker layers this TP has stubs in.
    nLayersWithStubs_ = countLayers();
  }

  //=== Count number of tracker layers a given list of stubs are in.
  //=== By default, consider both PS+2S modules, but optionally consider only the PS ones.

  unsigned int TP::countLayers(bool onlyPS) {
    //=== Unpack configuration parameters

    // Define layers using layer ID (true) or by bins in radius of 5 cm width (false).
    bool useLayerID = settings_->useLayerID();
    // When counting stubs in layers, actually histogram stubs in distance from beam-line with this bin size.
    float layerIDfromRadiusBin = settings_->layerIDfromRadiusBin();
    // Inner radius of tracker.
    float trackerInnerRadius = settings_->trackerInnerRadius();

    const int maxLayerID(30);
    std::vector<bool> foundLayers(maxLayerID, false);

    if (useLayerID) {
      // Count layers using CMSSW layer ID.
      for (const Stub& stub : assocStubs_) {
        if ((!onlyPS) || stub.psModule()) {  // Consider only stubs in PS modules if that option specified.
          int layerID = stub.layerId();
          if (layerID >= 0 && layerID < maxLayerID) {
            foundLayers[layerID] = true;
          } else {
            throw cms::Exception("Utility::invalid layer ID");
          }
        }
      }
    } else {
      // Count layers by binning stub distance from beam line.
      for (const Stub& stub : assocStubs_) {
        if ((!onlyPS) || stub.psModule()) {  // Consider only stubs in PS modules if that option specified.
          int layerID = (int)((stub.r() - trackerInnerRadius) / layerIDfromRadiusBin);
          if (layerID >= 0 && layerID < maxLayerID) {
            foundLayers[layerID] = true;
          } else {
            throw cms::Exception("Utility::invalid layer ID");
          }
        }
      }
    }

    unsigned int ncount = 0;
    for (const bool& found : foundLayers) {
      if (found)
        ncount++;
    }

    return ncount;
  }

  void TP::fillUseForVertexReco() {
    useForVertexReco_ = false;
    if (use_) {
      useForVertexReco_ = settings_->tpsUseForVtxReco()(trackingParticle_.get());
    }

    if (useForVertexReco_) {
      useForVertexReco_ = (countLayers() >= settings_->genMinStubLayers() and countLayers(true) >= 2);
    }
  }

  //=== Check if this tracking particle is worth keeping.
  void TP::fillUse() {
    // Use looser cuts here those those used for tracking efficiency measurement.
    // Keep only those TP that have a chance (allowing for finite track resolution) of being reconstructed as L1 tracks. L1 tracks not matching these TP will be defined as fake.
    use_ = settings_->tpsUse()(trackingParticle_.get());
  }

  //=== Check if this tracking particle can be used to measure the L1 tracking efficiency.
  void TP::fillUseForEff() {
    useForEff_ = false;
    if (use_) {
      useForEff_ = settings_->tpsUseForEff()(trackingParticle_.get());
    }
  }

  //=== Check if this tracking particle can be used to measure the L1 tracking algorithmic efficiency (makes stubs in enough layers).
  void TP::fillUseForAlgEff() {
    useForAlgEff_ = false;
    if (useForEff_) {
      useForAlgEff_ = (countLayers() >= settings_->genMinStubLayers());
    }
  }

}  // namespace l1tVertexFinder
