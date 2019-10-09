#ifndef CondFormats_GeometryObjects_CaloSimulationParameters_h
#define CondFormats_GeometryObjects_CaloSimulationParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

class CaloSimulationParameters {
public:
  CaloSimulationParameters(void) {}
  ~CaloSimulationParameters(void) {}

  std::vector<std::string> caloNames_;
  std::vector<int> levels_;
  std::vector<int> neighbours_;
  std::vector<std::string> insideNames_;
  std::vector<int> insideLevel_;

  std::vector<std::string> fCaloNames_;
  std::vector<int> fLevels_;

  COND_SERIALIZABLE;
};

#endif
