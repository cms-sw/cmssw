import FWCore.ParameterSet.Config as cms

# $Id: CaloTowersES_cfi.py,v 1.5 2008/07/11 14:35:26 rahatlou Exp $
#
# Event Setup necessary for CaloTowers reconstruction
#
#include "Geometry/CaloEventSetup/data/CaloTowerConstituents.cfi"
CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)


