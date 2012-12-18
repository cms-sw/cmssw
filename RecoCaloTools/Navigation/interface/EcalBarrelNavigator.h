#ifndef RECOCALOTOOLS_NAVIGATION_ECALBARRELNAVIGATOR_H
#define RECOCALOTOOLS_NAVIGATION_ECALBARRELNAVIGATOR_H 1

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"

using EcalBarrelNavigator = CaloNavigator<EBDetId>; 

using EcalBarrelNavigatorHT = CaloNavigator<EBDetId, EcalBarrelHardcodedTopology>; 

#endif
