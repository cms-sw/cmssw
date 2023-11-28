#include "Geometry/HcalCommonData/interface/CaloSimulationConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

CaloSimulationConstants::CaloSimulationConstants(const CaloSimulationParameters* csp) : calospar_(csp) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom")
      << "CaloSimulationConstants::CaloSimulationConstants (const CaloSimulationParameters* csp) constructor";
#endif
}

CaloSimulationConstants::~CaloSimulationConstants() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimulationConstants::destructed!!!";
#endif
}
