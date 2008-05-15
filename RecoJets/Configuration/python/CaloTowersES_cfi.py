import FWCore.ParameterSet.Config as cms

# $Id: CaloTowersES.cfi,v 1.4 2008/05/10 09:51:43 fedor Exp $
#
# Event Setup necessary for CaloTowers reconstruction
#
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
#include "Geometry/CaloEventSetup/data/CaloTowerConstituents.cfi"
CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)


