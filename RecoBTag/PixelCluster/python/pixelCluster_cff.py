import FWCore.ParameterSet.Config as cms

from RecoBTag.PixelCluster.pixelClusterTagInfos_cfi import *

pixelClusterTagInfosAK4PFJetsCHS = pixelClusterTagInfos.clone()
pixelClusterTagInfosAK4PFJetsCHS.jets = cms.InputTag("ak4PFJetsCHS")

pixelClusterTagInfosAK8PFJetsCHS = pixelClusterTagInfos.clone()
pixelClusterTagInfosAK8PFJetsCHS.jets = cms.InputTag("ak8PFJetsCHS")

pixelClusterTagInfosAK8PFJetsCHSSoftDrop = pixelClusterTagInfos.clone()
pixelClusterTagInfosAK8PFJetsCHSSoftDrop.jets = cms.InputTag("ak8PFJetsCHSSoftDrop")

pixelClusterTask = cms.Task(
    pixelClusterTagInfosAK4PFJetsCHS,
    pixelClusterTagInfosAK8PFJetsCHS,
    pixelClusterTagInfosAK8PFJetsCHSSoftDrop,
)
