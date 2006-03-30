#include "FWCore/Framework/interface/ModuleFactory.h"
#include "Geometry/CaloEventSetup/src/CaloGeometryBuilder.h"
#include "Geometry/CaloEventSetup/src/CaloTowerTopologyBuilder.h"
#include "Geometry/CaloEventSetup/src/CaloTopologyBuilder.h"

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CaloGeometryBuilder)
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CaloTowerTopologyBuilder)
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CaloTopologyBuilder)
