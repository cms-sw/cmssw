import FWCore.ParameterSet.Config as cms

ecalEndcapRawDataTask = cms.EDAnalyzer("EERawDataTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    FEDRawDataCollection = cms.InputTag("rawDataCollector"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker")
)
