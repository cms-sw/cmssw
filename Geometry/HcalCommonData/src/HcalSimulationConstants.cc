#include "Geometry/HcalCommonData/interface/HcalSimulationConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

HcalSimulationConstants::HcalSimulationConstants(const HcalSimulationParameters* hsp) : hspar_(hsp) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom")
      << "HcalSimulationConstants::HcalSimulationConstants (const HcalSimulationParameters* hsp) constructor";
#endif
}

HcalSimulationConstants::~HcalSimulationConstants() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalSimulationConstants::destructed!!!";
#endif
}
