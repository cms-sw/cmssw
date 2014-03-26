#include "FWCore/Framework/interface/ModuleFactory.h"
#include "Geometry/CaloEventSetup/plugins/CaloGeometryBuilder.h"
#include "Geometry/CaloEventSetup/plugins/ShashlikTopologyBuilder.h"
#include "Geometry/CaloEventSetup/plugins/CaloTowerConstituentsMapBuilder.h"
#include "Geometry/CaloEventSetup/plugins/EcalTrigTowerConstituentsMapBuilder.h"
#include "Geometry/CaloEventSetup/plugins/CaloTopologyBuilder.h"

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CaloGeometryBuilder);
DEFINE_FWK_EVENTSETUP_MODULE(CaloTowerConstituentsMapBuilder);
DEFINE_FWK_EVENTSETUP_MODULE(EcalTrigTowerConstituentsMapBuilder);
DEFINE_FWK_EVENTSETUP_MODULE(CaloTopologyBuilder);
DEFINE_FWK_EVENTSETUP_MODULE(ShashlikTopologyBuilder);
