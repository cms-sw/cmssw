import FWCore.ParameterSet.Config as cms

muonGEMDigis = cms.EDProducer("GEMRawToDigiModule",
    InputLabel = cms.InputTag("rawDataCollector"),
    mightGet = cms.optional.untracked.vstring,
    unPackStatusDigis = cms.bool(False),
    useDBEMap = cms.bool(False)
)
