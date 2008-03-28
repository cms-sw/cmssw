import FWCore.ParameterSet.Config as cms

from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# Calo geometry/topology services
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
#include "Geometry/CaloEventSetup/data/CaloTowerConstituents.cfi"
#es_module = CaloTowerConstituentsMapBuilder {
#    untracked string MapFile="Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz"
#}
from Geometry.CaloEventSetup.CaloTopology_cfi import *
#es_module = HcalTopologyIdealEP {}
from RecoParticleFlow.PFClusterProducer.towerMakerPF_cfi import *
from RecoParticleFlow.PFClusterProducer.caloTowersPF_cfi import *
caloTowersPFRec = cms.Sequence(towerMakerPF*caloTowersPF)

