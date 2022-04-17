#ifndef L1Trigger_L1THGCal_HGCalCluster_SA_h
#define L1Trigger_L1THGCal_HGCalCluster_SA_h

#include <vector>

namespace l1thgcfirmware {

  class HGCalCluster {
  public:
    HGCalCluster(float x,
                 float y,
                 float z,
                 int zside,
                 unsigned int layer,
                 float eta,
                 float phi,
                 float pt,
                 float mipPt,
                 unsigned int index_cmssw)
        : x_(x),
          y_(y),
          z_(z),
          zside_(zside),
          layer_(layer),
          eta_(eta),
          phi_(phi),
          pt_(pt),
          mipPt_(mipPt),
          index_cmssw_(index_cmssw) {}

    ~HGCalCluster() = default;

    float x() const { return x_; }
    float y() const { return y_; }
    float z() const { return z_; }
    float zside() const { return zside_; }
    unsigned int layer() const { return layer_; }
    float eta() const { return eta_; }
    float phi() const { return phi_; }
    float pt() const { return pt_; }
    float mipPt() const { return mipPt_; }
    unsigned int index_cmssw() const { return index_cmssw_; }

  private:
    float x_;
    float y_;
    float z_;
    int zside_;
    unsigned int layer_;
    float eta_;
    float phi_;
    float pt_;
    float mipPt_;
    unsigned int index_cmssw_;
  };

  typedef std::vector<HGCalCluster> HGCalClusterSACollection;

}  // namespace l1thgcfirmware

#endif
