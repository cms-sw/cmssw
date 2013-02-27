import FWCore.ParameterSet.Config as cms

ecalBarrelRawDataTask = cms.EDAnalyzer("EBRawDataTask",
    prefixME = cms.untracked.string('EcalBarrel'),
                                       subfolder = cms.untracked.string(''),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    FEDRawDataCollection = cms.InputTag("rawDataCollector"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker")
)
