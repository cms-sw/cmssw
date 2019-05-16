#ifndef CalibCalorimetry_HBHERecalibration_h
#define CalibCalorimetry_HBHERecalibration_h

#include "CondFormats/HcalObjects/interface/HBHEDarkening.h"

#include <vector>
#include <string>

// Simple recalibration algorithm for radiation damage to HB and HE
// produces response correction for a depth based on average of darkening per layer, weighted by mean energy per layer
// (a depth can contain several layers)
// (mean energy per layer derived from 50 GeV single pion scan in MC)

class HBHERecalibration {
public:
  HBHERecalibration(float intlumi, float cutoff, std::string meanenergies);
  ~HBHERecalibration();

  //accessors
  float getCorr(int ieta, int depth) const;
  void setup(const std::vector<std::vector<int>>& m_segmentation, const HBHEDarkening* darkening);
  int maxDepth() const { return max_depth_; }

private:
  //helper
  void initialize();

  //members
  float intlumi_;
  float cutoff_;
  int ieta_shift_;
  int max_depth_;
  std::vector<std::vector<float>> meanenergies_;
  const HBHEDarkening* darkening_;
  std::vector<std::vector<int>> dsegm_;
  std::vector<std::vector<float>> corr_;
};

#endif  // HBHERecalibration_h
