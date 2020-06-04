#include "Geometry/HcalCommonData/interface/HcalDDDSimulationConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

HcalDDDSimulationConstants::HcalDDDSimulationConstants(const HcalSimulationParameters* hsp) : hspar_(hsp) {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HCalGeom")
      << "HcalDDDSimulationConstants::HcalDDDSimulationConstants (const HcalSimulationParameters* hsp) constructor\n";
#endif
}

HcalDDDSimulationConstants::~HcalDDDSimulationConstants() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HCalGeom") << "HcalDDDSimulationConstants::destructed!!!\n";
#endif
}
