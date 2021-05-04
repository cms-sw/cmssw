#ifndef CondFormats_GeometryObjects_EcalSimulationParameters_h
#define CondFormats_GeometryObjects_EcalSimulationParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

class EcalSimulationParameters {
public:
  EcalSimulationParameters(void) = default;
  ~EcalSimulationParameters(void) = default;

  int nxtalEta_;
  int nxtalPhi_;
  int phiBaskets_;
  std::vector<int> etaBaskets_;
  int ncrys_;
  int nmods_;
  bool useWeight_;
  std::string depth1Name_;
  std::string depth2Name_;
  std::vector<std::string> lvNames_;
  std::vector<std::string> matNames_;
  std::vector<double> dzs_;

  COND_SERIALIZABLE;
};

#endif
