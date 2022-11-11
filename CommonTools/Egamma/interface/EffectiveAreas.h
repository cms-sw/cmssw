#ifndef CommonTools_Egamma_EffectiveAreas_h
#define CommonTools_Egamma_EffectiveAreas_h

#include <vector>
#include <string>

class EffectiveAreas {
public:
  // Constructor, destructor
  EffectiveAreas(const std::string& filename, const bool quadraticEAflag = false);

  // Accessors
  const float getEffectiveArea(float eta) const;
  const float getLinearEA(float eta) const;
  const float getQuadraticEA(float eta) const;

  // Utility functions
  void printEffectiveAreas() const;
  void checkConsistency() const;

private:
  // Data members
  const std::string filename_;              // effective areas source file name
  std::vector<float> absEtaMin_;            // low limit of the eta range
  std::vector<float> absEtaMax_;            // upper limit of the eta range
  std::vector<float> effectiveAreaValues_;  // effective area for this eta range

  // Following members are for quadratic PU-correction (introduced for cutBasedPhotonID in Run3_122X)
  const bool quadraticEAflag_;
  std::vector<float> linearEffectiveAreaValues_;
  std::vector<float> quadraticEffectiveAreaValues_;
};

#endif
