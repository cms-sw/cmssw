import FWCore.ParameterSet.Config as cms

# $Id: CaloTowersES_cfi.py,v 1.4 2008/05/19 20:56:52 rpw Exp $
#
# Event Setup necessary for CaloTowers reconstruction
#
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
#include "Geometry/CaloEventSetup/data/CaloTowerConstituents.cfi"
CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)


