#ifndef CalibCalorimetry_HcalAlgos_HcalPulseContainmentManager_h
#define CalibCalorimetry_HcalAlgos_HcalPulseContainmentManager_h

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentCorrection.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "FWCore/Framework/interface/ESHandle.h"

class HcalPulseContainmentManager {
public:
  HcalPulseContainmentManager(float  max_fracerror);
  double correction(const HcalDetId & detId, int toAdd, float fixedphase_ns, double fc_ampl);
  const HcalPulseContainmentCorrection * get(const HcalDetId & detId, int toAdd, float fixedphase_ns);

  void beginRun(edm::EventSetup const & es);
  void beginRun(const HcalDbService* conditions, const HcalTimeSlew* delay);

  void setTimeSlew(const HcalTimeSlew* timeSlew) {
    hcalTimeSlew_delay_ = timeSlew;
  }
private:

  struct HcalPulseContainmentEntry {
    HcalPulseContainmentEntry(int toAdd, float fixedphase_ns, const HcalPulseShape * shape,  const HcalPulseContainmentCorrection & correction)
      : toAdd_(toAdd), fixedphase_ns_(fixedphase_ns),shape_(shape), correction_(correction) {}
    int toAdd_;
    float fixedphase_ns_;
    const HcalPulseShape * shape_;
    HcalPulseContainmentCorrection correction_;
  };

  std::vector<HcalPulseContainmentEntry> entries_;
  HcalPulseShapes shapes_;
  float fixedphase_ns_;
  float max_fracerror_;

  const HcalTimeSlew* hcalTimeSlew_delay_;
};

#endif
