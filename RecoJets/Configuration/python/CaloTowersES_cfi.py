import FWCore.ParameterSet.Config as cms

# $Id: CaloTowersES_cfi.py,v 1.6 2008/07/17 14:28:36 rahatlou Exp $
#
# Event Setup necessary for CaloTowers reconstruction
#
#include "Geometry/CaloEventSetup/data/CaloTowerConstituents.cfi"
CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)


