#ifndef RecoEgamma_EgammaTools_EffectiveAreas_H
#define RecoEgamma_EgammaTools_EffectiveAreas_H

#include <vector>

#include <string>

class EffectiveAreas {

public:
  // Constructor, destructor
  EffectiveAreas(const std::string& filename);

  // Accessors
  const float getEffectiveArea(float eta) const;

  // Utility functions
  void printEffectiveAreas() const;
  void checkConsistency() const;

private:
  // Data members
  const std::string  filename_;  // effective areas source file name
  std::vector<float> absEtaMin_; // low limit of the eta range
  std::vector<float> absEtaMax_; // upper limit of the eta range
  std::vector<float> effectiveAreaValues_; // effective area for this eta range

};

#endif
