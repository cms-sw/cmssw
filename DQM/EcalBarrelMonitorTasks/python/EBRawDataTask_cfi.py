import FWCore.ParameterSet.Config as cms

ecalBarrelRawDataTask = cms.EDAnalyzer("EBRawDataTask",
    prefixME = cms.untracked.string('Ecal'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    FEDRawDataCollection = cms.InputTag("rawDataCollector"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker")
)
