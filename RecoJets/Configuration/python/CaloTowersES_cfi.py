import FWCore.ParameterSet.Config as cms

import Geometry.CaloEventSetup.caloTowerConstituents_cfi

CaloTowerConstituentsMapBuilder = Geometry.CaloEventSetup.caloTowerConstituents_cfi.caloTowerConstituents.clone()
CaloTowerConstituentsMapBuilder.MapFile = "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz"

## import Geometry.HcalEventSetup.hcalTopologyConstants_cfi as hcalTopologyConstants_cfi
## caloTowerConstituentsMapBuilder.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)

# $Id: CaloTowersES_cfi.py,v 1.7 2012/08/28 14:26:18 yana Exp $
#
# Event Setup necessary for CaloTowers reconstruction
#
#include "Geometry/CaloEventSetup/data/CaloTowerConstituents.cfi"
## CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
##     MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
## )


