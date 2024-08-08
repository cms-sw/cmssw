import FWCore.ParameterSet.Config as cms

hltMuonGEMDigis = cms.EDProducer("GEMRawToDigiModule",
    InputLabel = cms.InputTag("rawDataCollector"),
    mightGet = cms.optional.untracked.vstring,
    useDBEMap = cms.bool(False)
)
