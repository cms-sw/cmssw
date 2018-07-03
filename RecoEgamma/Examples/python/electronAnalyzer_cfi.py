import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
electronAnalyzer = DQMEDAnalyzer('ElectronAnalyzer',
    outputFile = cms.string('electronAnalysisOutput.root'),
    REleCut = cms.double(0.1),
    mcProducer = cms.string('source'),
    superClusterProducer = cms.InputTag("hybridSuperClusters",""),
    minElePt = cms.double(5.0),
    electronProducer = cms.InputTag("pixelMatchElectrons"),
)


