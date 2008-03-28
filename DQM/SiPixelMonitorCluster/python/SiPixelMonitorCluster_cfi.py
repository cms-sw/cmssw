import FWCore.ParameterSet.Config as cms

SiPixelClusterSource = cms.EDFilter("SiPixelClusterSource",
    src = cms.InputTag("siPixelClusters"),
    outputFile = cms.string('Pixel_DQM_Cluster.root'),
    saveFile = cms.untracked.bool(False)
)


