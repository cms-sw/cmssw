import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.likelihoodPdfsDB_cfi import *
from RecoEgamma.ElectronIdentification.likelihoodESetup_cfi import *
eidLikelihood = cms.EDFilter("EleIdLikelihoodRef",
    filter = cms.bool(False),
    src = cms.InputTag("pixelMatchGsfElectrons"),
    endcapClusterShapeAssociation = cms.InputTag("islandBasicClusters","islandEndcapShapeAssoc"),
    threshold = cms.double(0.5),
    barrelClusterShapeAssociation = cms.InputTag("hybridSuperClusters","hybridShapeAssoc")
)


