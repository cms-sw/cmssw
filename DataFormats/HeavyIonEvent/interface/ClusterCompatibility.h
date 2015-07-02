#ifndef DataFormats_ClusterCompatibility_h
#define DataFormats_ClusterCompatibility_h

#include <vector>

namespace reco { class ClusterCompatibility {
public:

  ClusterCompatibility();
  virtual ~ClusterCompatibility();

  /// Number of valid pixel clusters
  int nValidPixelHits() const { return nValidPixelHits_; }

  /// Number of vertex-position hypotheses tested
  int size() const { return z0_.size(); }

  /// Vertex z position for the i-th vertex-position hypothesis
  float z0(int i) const { return z0_[i]; }

  /// Number of compatible non-edge pixel-barrel clusters 
  /// for the i-th vertex-position hypothesis
  int nHit(int i) const { return nHit_[i]; }

  /// Sum of the difference between the expected and actual 
  /// width of all compatible non-edge pixel-barrel clusters 
  /// for the i-th vertex-position hypothesis
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
