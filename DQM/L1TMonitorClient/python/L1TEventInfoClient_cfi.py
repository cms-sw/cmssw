import FWCore.ParameterSet.Config as cms

l1tEventInfoClient = cms.EDAnalyzer("L1TEventInfoClient",
    prescaleLS = cms.untracked.int32(-1),
    monitorDir = cms.untracked.string(''),
    prescaleEvt = cms.untracked.int32(1),
    dataMaskedSystems = cms.untracked.vstring("empty"),
    emulMaskedSystems = cms.untracked.vstring("all"),
    thresholdLS = cms.untracked.int32(5),
    GCT_NonIsoEm_threshold = cms.untracked.double(100000),
    GCT_IsoEm_threshold = cms.untracked.double(100000),
    GCT_TauJets_threshold = cms.untracked.double(100000),
    GCT_AllJets_threshold = cms.untracked.double(100000),
    GMT_Muons_threshold = cms.untracked.double(100000)

)


