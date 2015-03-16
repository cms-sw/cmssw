import FWCore.ParameterSet.Config as cms

bxTiming = cms.EDAnalyzer("BxTiming",
    HistFolder = cms.untracked.string('L1T/BXSynch'),
    VerboseFlag = cms.untracked.int32(0),
    HistFile = cms.untracked.string(''),
    DQMStore = cms.untracked.bool(True),
    GtBitList = cms.untracked.vint32(0, 1),
    ReferenceFedId = cms.untracked.int32(1024),
    GtSource = cms.untracked.InputTag("gtDigis"),
    FedSource = cms.untracked.InputTag("rawDataCollector"),
    RunInFilterFarm = cms.untracked.bool(False)
)


