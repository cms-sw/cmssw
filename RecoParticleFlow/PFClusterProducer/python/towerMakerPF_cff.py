import FWCore.ParameterSet.Config as cms

# Calo geometry/topology services
#include "Geometry/CaloEventSetup/data/CaloTowerConstituents.cfi"
#es_module = CaloTowerConstituentsMapBuilder {
#    untracked string MapFile="Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz"
#}
from Geometry.CaloEventSetup.CaloTopology_cfi import *
#es_module = HcalTopologyIdealEP {}
from RecoParticleFlow.PFClusterProducer.towerMakerPF_cfi import *
from RecoParticleFlow.PFClusterProducer.caloTowersPF_cfi import *
caloTowersPFRec = cms.Sequence(towerMakerPF*caloTowersPF)

