#include "Geometry/HcalCommonData/interface/CaloSimulationConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

CaloSimulationConstants::CaloSimulationConstants(const CaloSimulationParameters* csp) : calospar_(csp) {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HCalGeom")
      << "CaloSimulationConstants::CaloSimulationConstants (const CaloSimulationParameters* csp) constructor\n";
#endif
}

CaloSimulationConstants::~CaloSimulationConstants() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HCalGeom") << "CaloSimulationConstants::destructed!!!\n";
#endif
}
