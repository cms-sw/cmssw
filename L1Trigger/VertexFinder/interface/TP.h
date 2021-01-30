#ifndef __L1Trigger_VertexFinder_TP_h__
#define __L1Trigger_VertexFinder_TP_h__

#include "DataFormats/Common/interface/Ptr.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "L1Trigger/VertexFinder/interface/AnalysisSettings.h"
#include "L1Trigger/VertexFinder/interface/Stub.h"
#include "L1Trigger/VertexFinder/interface/utility.h"

namespace l1tVertexFinder {

  typedef edm::Ptr<TrackingParticle> TrackingParticlePtr;

  class AnalysisSettings;
  class Stub;

  class TP : public TrackingParticlePtr {
  public:
    // Fill useful info about tracking particle.
    TP(TrackingParticlePtr tpPtr, const AnalysisSettings& settings);
    ~TP() {}

    // Fill truth info with association from tracking particle to stubs.
    void setMatchingStubs(const std::vector<const Stub*>& vMatchingStubs);

    // == Functions for returning info about tracking particles ===

    // Did it come from the main physics collision or from pileup?
    bool physicsCollision() const { return physicsCollision_; }
    // Functions returning stubs produced by tracking particle.
    unsigned int numAssocStubs() const { return assocStubs_.size(); }
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

    // Calculate how many tracker layers this TP has stubs in.
    void calcNumLayers() { nLayersWithStubs_ = utility::countLayers(*settings_, assocStubs_); }

  private:
    const AnalysisSettings* settings_;  // Configuration parameters

    bool inTimeBx_;          // TP came from in-time bunch crossing.
    bool physicsCollision_;  // True if TP from physics collision rather than pileup.
    bool use_;               // TP is worth keeping (e.g. for fake rate measurement)
    bool useForEff_;         // TP can be used for tracking efficiency measurement.
    bool useForAlgEff_;      // TP can be used for tracking algorithmic efficiency measurement.
    bool useForVertexReco_;

    std::vector<const Stub*> assocStubs_;
    unsigned int nLayersWithStubs_;  // Number of tracker layers with stubs from this TP.
  };

}  // end namespace l1tVertexFinder

#endif
