#ifndef CalibCalorimetry_HcalAlgos_HcalPulseContainmentManager_h
#define CalibCalorimetry_HcalAlgos_HcalPulseContainmentManager_h

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentCorrection.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalPulseContainmentManager {
public:
  HcalPulseContainmentManager(float fixedphase_ns, float  max_fracerror);
  double correction(const HcalDetId & detId, int toAdd, double fc_ampl);
  const HcalPulseContainmentCorrection * get(const HcalDetId & detId, int toAdd);

private:

  struct HcalPulseContainmentEntry {
    HcalPulseContainmentEntry(int toAdd, const HcalPulseShape * shape, 
                              const HcalPulseContainmentCorrection & correction)
    : toAdd_(toAdd), shape_(shape), correction_(correction) {}
    int toAdd_;
    const HcalPulseShape * shape_;
    HcalPulseContainmentCorrection correction_;
  };

  std::vector<HcalPulseContainmentEntry> entries_;
  // indexed on the dense HcalDetId, and stores an index into entries_;
  std::vector<short> denseIndexToEntry_;
  HcalPulseShapes shapes_;
  float fixedphase_ns_;
  float max_fracerror_;
};

#endif
