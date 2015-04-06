#ifndef EffectiveAreas_H
#define EffectiveAreas_H

#include <vector>

#include <TString.h>

class EffectiveAreas {

public:
  // Constructor, destructor
  EffectiveAreas(TString filename);
  ~EffectiveAreas();

  // Accessors
  const float getEffectiveArea(float eta);

  // Utility functions
  void printEffectiveAreas();
  void checkConsistency();

private:
  // Data members
  TString            filename_;        // effective areas source file name
  std::vector<float> absEtaMin_; // low limit of the eta range
  std::vector<float> absEtaMax_; // upper limit of the eta range
  std::vector<float> effectiveAreaValues_; // effective area for this eta range

};

#endif
