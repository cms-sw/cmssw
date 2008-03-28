import FWCore.ParameterSet.Config as cms

electronAnalyzer = cms.EDAnalyzer("ElectronAnalyzer",
    outputFile = cms.string('electronAnalysisOutput.root'),
    REleCut = cms.double(0.1),
    islandBarrelBasicClusterShapes = cms.string('islandBarrelShape'),
    mcProducer = cms.string('source'),
    # InputTag superClusterProducer = hybridSuperClusters
    superClusterProducer = cms.InputTag("islandSuperClusters","islandBarrelSuperClusters"),
    minElePt = cms.double(5.0),
    electronProducer = cms.InputTag("pixelMatchElectrons"),
    islandBarrelBasicClusterProducer = cms.string('islandBasicClusters'),
    islandBarrelBasicClusterCollection = cms.string('islandBarrelBasicClusters')
)


