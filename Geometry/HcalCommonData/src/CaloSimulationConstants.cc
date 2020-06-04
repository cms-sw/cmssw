#include "Geometry/HcalCommonData/interface/CaloDDDSimulationConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

CaloDDDSimulationConstants::CaloDDDSimulationConstants(const CaloSimulationParameters* csp) : calospar_(csp) {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HCalGeom")
      << "CaloDDDSimulationConstants::CaloDDDSimulationConstants (const CaloSimulationParameters* csp) constructor\n";
#endif
}

CaloDDDSimulationConstants::~CaloDDDSimulationConstants() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HCalGeom") << "CaloDDDSimulationConstants::destructed!!!\n";
#endif
}
