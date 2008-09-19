#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

EVENTSETUP_DATA_REG(HcalTopology);
EVENTSETUP_DATA_REG(CaloTowerConstituentsMap);
EVENTSETUP_DATA_REG(EcalTrigTowerConstituentsMap);
