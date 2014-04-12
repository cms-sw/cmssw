import FWCore.ParameterSet.Config as cms

dtpacker = cms.EDProducer("DTDigiToRawModule",
    useStandardFEDid = cms.untracked.bool(True),
    debugMode = cms.untracked.bool(False),
    digiColl = cms.InputTag("simMuonDTDigis"),
    minFEDid = cms.untracked.int32(770),
    maxFEDid = cms.untracked.int32(775)
)
