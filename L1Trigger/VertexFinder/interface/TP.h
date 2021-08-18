#ifndef __L1Trigger_VertexFinder_TP_h__
#define __L1Trigger_VertexFinder_TP_h__

#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/VertexFinder/interface/AnalysisSettings.h"
#include "L1Trigger/VertexFinder/interface/Stub.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

namespace l1tVertexFinder {

  class AnalysisSettings;
  class Stub;

  typedef edm::Ptr<TrackingParticle> TrackingParticlePtr;

  class TP {
  public:
    // Fill useful info about tracking particle.
    TP();
    TP(const TrackingParticlePtr& tpPtr, const AnalysisSettings& settings);
    ~TP() {}

    // Need the operator== to compare 2 TP types.
    bool operator==(const TP& rhs) const {
      return (trackingParticle_ == rhs.trackingParticle_ && settings_ == rhs.settings_ && inTimeBx_ == rhs.inTimeBx_ &&
              physicsCollision_ == rhs.physicsCollision_ && use_ == rhs.use_ && useForEff_ == rhs.useForEff_ &&
              useForAlgEff_ == rhs.useForAlgEff_ && useForVertexReco_ == rhs.useForVertexReco_ &&
              nLayersWithStubs_ == rhs.nLayersWithStubs_ && assocStubs_ == rhs.assocStubs_);
    }

    // Fill truth info with association from tracking particle to stubs.
    void setMatchingStubs(const std::vector<Stub>& vMatchingStubs);

    // == Functions for returning info about tracking particles ===

    // Count number of tracker layers a given list of stubs are in.
    // By default, considers both PS+2S modules, but optionally considers only the PS ones if onlyPS = true.
    unsigned int countLayers(bool onlyPS = false);
    // Return a C++ pointer to the tracking particle
    const TrackingParticlePtr getTrackingParticle() const { return trackingParticle_; }
    // Did it come from the main physics collision or from pileup?
    bool physicsCollision() const { return physicsCollision_; }
    // Functions returning stubs produced by tracking particle.
    unsigned int numAssocStubs() const { return assocStubs_.size(); }
    // Overload operator-> to return a C++ pointer to the underlying TrackingParticle
    const TrackingParticlePtr operator->() const { return trackingParticle_; }
    // TP is worth keeping (e.g. for fake rate measurement)
    bool use() const { return use_; }
    // TP can be used to measure the L1 tracking efficiency
    bool useForEff() const { return useForEff_; }
    // TP can be used for algorithmic efficiency measurement (also requires stubs in enough layers).
    bool useForAlgEff() const { return useForAlgEff_; }
    // TP can be used for vertex reconstruction measuremetn
    bool useForVertexReco() const { return useForVertexReco_; }

  private:
    void fillUse();           // Fill the use_ flag.
    void fillUseForEff();     // Fill the useForEff_ flag.
    void fillUseForAlgEff();  // Fill the useforAlgEff_ flag.
    void fillUseForVertexReco();

  private:
    TrackingParticlePtr trackingParticle_;  // Underlying TrackingParticle pointer
    const AnalysisSettings* settings_;      // Configuration parameters

    bool inTimeBx_;          // TP came from in-time bunch crossing.
    bool physicsCollision_;  // True if TP from physics collision rather than pileup.
    bool use_;               // TP is worth keeping (e.g. for fake rate measurement)
    bool useForEff_;         // TP can be used for tracking efficiency measurement.
    bool useForAlgEff_;      // TP can be used for tracking algorithmic efficiency measurement.
    bool useForVertexReco_;

    std::vector<Stub> assocStubs_;
    unsigned int nLayersWithStubs_;  // Number of tracker layers with stubs from this TP.
  };

}  // end namespace l1tVertexFinder

#endif
