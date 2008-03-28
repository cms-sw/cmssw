import FWCore.ParameterSet.Config as cms

# $Id: CaloTowersES.cfi,v 1.1 2008/03/06 16:10:29 fedor Exp $
#
# Event Setup necessary for CaloTowers reconstruction
#
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
from Geometry.CaloEventSetup.CaloTopology_cfi import *
HcalTopologyIdealEP = cms.ESProducer("HcalTopologyIdealEP")

#include "Geometry/CaloEventSetup/data/CaloTowerConstituents.cfi"
CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)


