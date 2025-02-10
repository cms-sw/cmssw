import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

l1tCaloLayer1Summary = DQMEDAnalyzer("L1TCaloLayer1Summary",
    caloLayer1CICADAScore = cms.InputTag("caloLayer1Digis", "CICADAScore"),
    gtCICADAScore = cms.InputTag("gtStage2Digis", "CICADAScore"),
    simCICADAScore = cms.InputTag("dqmSimCaloStage2Layer1Summary", "CICADAScore"),
    caloLayer1Regions = cms.InputTag("caloLayer1Digis", ""),
    simRegions = cms.InputTag("dqmSimCaloStage2Layer1Digis", ""),
    fedRawDataLabel  = cms.InputTag("rawDataCollector"),
    histFolder = cms.string('L1T/L1TCaloLayer1Summary')
)
