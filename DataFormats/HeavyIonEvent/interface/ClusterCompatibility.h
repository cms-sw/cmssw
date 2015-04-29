#ifndef DataFormats_ClusterCompatibility_h
#define DataFormats_ClusterCompatibility_h

#include <vector>

namespace reco { class ClusterCompatibility {
public:

  ClusterCompatibility();
  virtual ~ClusterCompatibility();

  int nValidPixelHits() const { return nValidPixelHits_; }

  int size() const { return z0_.size(); }
  float z0(int i) const { return z0_[i]; }
  int nHit(int i) const { return nHit_[i]; }
  float chi(int i) const { return chi_[i]; }

  void append(float, int, float);
  void setNValidPixelHits(int nPxl) { nValidPixelHits_ = nPxl; }

protected:
 

  int nValidPixelHits_;
 
  std::vector<float> z0_;
  std::vector<int> nHit_;
  std::vector<float> chi_;

};

}
#endif
