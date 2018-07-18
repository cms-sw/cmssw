#ifndef __PhysicsTools_PatAlgos_MuonMvaEstimator__
#define __PhysicsTools_PatAlgos_MuonMvaEstimator__

#include "DataFormats/BTauReco/interface/JetTag.h"

#include <memory>
#include <string>

class GBRForest;

namespace pat {
  class Muon;
}

namespace reco {
  class JetCorrector;
  class Vertex;
}

namespace pat {
  class MuonMvaEstimator{
  public:

    MuonMvaEstimator(const std::string& weightsfile, float dRmax);

    ~MuonMvaEstimator();

    float computeMva(const pat::Muon& imuon,
                     const reco::Vertex& vertex,
                     const reco::JetTagCollection& bTags,
                     float& jetPtRatio,
                     float& jetPtRel,
                     const reco::JetCorrector* correctorL1=nullptr,
                     const reco::JetCorrector* correctorL1L2L3Res=nullptr) const;

  private:

    std::unique_ptr<const GBRForest> gbrForest_;
    float dRmax_;

    /// MVA VAriables
    float pt_ = 0.0;
    float eta_ = 0.0;
    float jetNDauCharged_ = 0.0;
    float miniRelIsoCharged_ = 0.0;
    float miniRelIsoNeutral_ = 0.0;
    float jetPtRel_ = 0.0;
    float jetPtRatio_ = 0.0;
    float jetBTagCSV_ = 0.0;
    float sip_ = 0.0;
    float log_abs_dxyBS_ = 0.0;
    float log_abs_dzPV_ = 0.0;
    float segmentCompatibility_ = 0.0;
  };
}
#endif
