import FWCore.ParameterSet.Config as cms

hltMuonGEMDigis = cms.EDProducer("GEMRawToDigiModule",
    InputLabel = cms.InputTag("rawDataCollector"),
    readMultiBX = cms.bool(True),
    mightGet = cms.optional.untracked.vstring,
    useDBEMap = cms.bool(False)
)
