#ifndef __PhysicsTools_PatAlgos_SoftMuonMvaEstimator__
#define __PhysicsTools_PatAlgos_SoftMuonMvaEstimator__

#include <memory>
#include <string>

class GBRForest;

namespace pat {
  class Muon;
}

namespace pat {
  class SoftMuonMvaEstimator{
  public:

    SoftMuonMvaEstimator(const std::string& weightsfile);

    ~SoftMuonMvaEstimator();

    float computeMva(const pat::Muon& imuon) const;

  private:

    std::unique_ptr<const GBRForest> gbrForest_;

    // MVA VAriables
    float segmentCompatibility_ = 0.0;
    float chi2LocalMomentum_ = 0.0;
    float chi2LocalPosition_ = 0.0;
    float glbTrackProbability_ = 0.0;
    float iValidFraction_ = 0.0;
    float layersWithMeasurement_ = 0.0;
    float trkKink_ = 0.0;
    float log2PlusGlbKink_ = 0.0;
    float timeAtIpInOutErr_ = 0.0;
    float outerChi2_ = 0.0;
    float innerChi2_ = 0.0;
    float trkRelChi2_ = 0.0;
    float vMuonHitComb_ = 0.0;
    float qProd_ = 0.0;

    // MVA Spectator
    float pID_ = 0.0;
    float pt_ = 0.0;
    float eta_ = 0.0;
    float momID_ = 0.0;

  };
}
#endif
