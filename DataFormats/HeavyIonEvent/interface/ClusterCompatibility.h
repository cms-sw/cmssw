#ifndef DataFormats_ClusterCompatibility_h
#define DataFormats_ClusterCompatibility_h

#include <vector>

namespace reco { class ClusterCompatibility {
public:

  ClusterCompatibility();
  ClusterCompatibility(float z0, int nHit, float chi);
  virtual ~ClusterCompatibility();

  float z0() const { return z0_; }
  int nHit() const { return nHit_; }
  float chi() const { return chi_; }

protected:
  
  float z0_;
  int nHit_;
  float chi_;

};

typedef std::vector<reco::ClusterCompatibility> ClusterCompatibilityCollection;

}

#endif
