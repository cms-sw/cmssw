#ifndef CondFormats_GeometryObjects_HcalSimulationParameters_h
#define CondFormats_GeometryObjects_HcalSimulationParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

class HcalSimulationParameters {
public:
  HcalSimulationParameters(void) {}
  ~HcalSimulationParameters(void) {}

  std::vector<double> attenuationLength_;
  std::vector<int> lambdaLimits_;
  std::vector<double> shortFiberLength_;
  std::vector<double> longFiberLength_;

  std::vector<int> pmtRight_;
  std::vector<int> pmtFiberRight_;
  std::vector<int> pmtLeft_;
  std::vector<int> pmtFiberLeft_;

  std::vector<int> hfLevels_;
  std::vector<std::string> hfNames_;
  std::vector<std::string> hfFibreNames_;
  std::vector<std::string> hfPMTNames_;
  std::vector<std::string> hfFibreStraightNames_;
  std::vector<std::string> hfFibreConicalNames_;
  std::vector<std::string> hcalMaterialNames_;

  COND_SERIALIZABLE;
};

#endif
