#include "Geometry/HcalEventSetup/interface/HcalHardcodeGeometryEP.h"
#include "Geometry/HcalEventSetup/src/CaloTowerHardcodeGeometryEP.h"
#include "Geometry/HcalEventSetup/interface/HcalTopologyIdealEP.h"
#include "Geometry/HcalEventSetup/interface/HcalDDDGeometryEP.h"
#include "Geometry/HcalEventSetup/interface/HcalAlignmentEP.h"
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalHardcodeGeometryEP);
DEFINE_FWK_EVENTSETUP_MODULE(CaloTowerHardcodeGeometryEP);
DEFINE_FWK_EVENTSETUP_MODULE(HcalTopologyIdealEP);
DEFINE_FWK_EVENTSETUP_MODULE(HcalDDDGeometryEP);
DEFINE_FWK_EVENTSETUP_MODULE(HcalAlignmentEP);
