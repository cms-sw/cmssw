#ifndef DataFormats_PatCandidates_SuperCluster_h
#define DataFormats_PatCandidates_SuperCluster_h

/***

This is a slimmed data format to be able to store all EGamma superclusters 
in MiniAOD as well as the necessary ID variables for EG workflows

The goal is to catch the superclusters that dont make it to electrons/photons (of which there are many) to enable efficiency measurements of this stage. 

So initially it was thought to make this a LeafCandidate but a 
supercluster is really a point + energy not a momentum so while more awkward not to just treat it as a p4, its more accurate to treat it as a position + energy

author: Sam Harper, Swagata Mukherjee

***/

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include <vector>

namespace reco {
  class SuperCluster;
}

namespace reco {

  class SlimmedSuperCluster {
  public:
    SlimmedSuperCluster() : rawEnergy_(0.), preshowerEnergy_(0.), trkIso_(0.) {}
    SlimmedSuperCluster(const reco::SuperCluster&, float trkIso = 0.);
    float rawEnergy() const { return rawEnergy_; }
    float preshowerEnergy() const { return preshowerEnergy_; }
    DetId seedId() const { return clusterSeedIds_.empty() != 0 ? DetId(0) : clusterSeedIds_.front(); }
    const std::vector<DetId>& clusterSeedIds() const { return clusterSeedIds_; }
    math::XYZPoint position() const;
    float trkIso() const { return trkIso_; }
    void setTrkIso(float val) { trkIso_ = val; }

  private:
    float correctedEnergy_;
    float rawEnergy_;
    float preshowerEnergy_;
    float rho_;
    float eta_;
    float phi_;
    std::vector<DetId> clusterSeedIds_;  //the first one is always the seed
    float trkIso_;
  };
}  // namespace reco
#endif
