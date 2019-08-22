#ifndef RecoEgamma_EgammaTools_LongDeps_h
#define RecoEgamma_EgammaTools_LongDeps_h
#include <vector>
#include <set>

/*
 * Stores information about the HGCal EM MultiCluster computed
 * using EGammaPCAHelper, including in particular hits restricted to
 * a cylinder along the shower axis with some given radius
 */

namespace hgcal {

  class LongDeps {
  public:
    LongDeps(float radius,
             const std::vector<float>& energyPerLayer,
             float energyEE,
             float energyFH,
             float energyBH,
             const std::set<int>& layers);
    ~LongDeps() {}
    // to check the radius used
    inline float radius() const { return radius_; };
    inline float energyEE() const { return energyEE_; }
    inline float energyFH() const { return energyFH_; }
    inline float energyBH() const { return energyBH_; }
    inline const std::vector<float>& energyPerLayer() const { return energyPerLayer_; }
    inline const std::set<int>& layers() const { return layers_; }
    inline unsigned nLayers() const { return layers_.size(); }
    inline int firstLayer() const { return (nLayers() > 0 ? *layers_.begin() : -1); }
    inline int lastLayer() const { return (nLayers() > 0 ? *layers_.rbegin() : -1); }
    inline int layerEfrac10() const { return lay_Efrac10_; }
    inline int layerEfrac90() const { return lay_Efrac90_; }
    inline float e4oEtot() const { return e4oEtot_; }

  private:
    std::vector<float> energyPerLayer_;
    float radius_;
    float energyEE_;
    float energyFH_;
    float energyBH_;
    std::set<int> layers_;
    int lay_Efrac10_;
    int lay_Efrac90_;
    float e4oEtot_;
  };

}  // namespace hgcal

#endif
