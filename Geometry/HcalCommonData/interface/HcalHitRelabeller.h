#ifndef SimCalorimetry_HcalSimProducers_HcalHitRelabeller_h
#define SimCalorimetry_HcalSimProducers_HcalHitRelabeller_h 1

#include <vector>
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalHitRelabeller {
public:
  HcalHitRelabeller(bool nd = false);
  void process(std::vector<PCaloHit>& hcalHits);
  void setGeometry(const HcalDDDRecConstants*&);
  DetId relabel(const uint32_t testId) const;
  static DetId relabel(const uint32_t testId, const HcalDDDRecConstants* theRecNumber);
  double energyWt(const uint32_t testId) const;

private:
  const HcalDDDRecConstants* theRecNumber;
  bool neutralDensity_;
};
#endif
