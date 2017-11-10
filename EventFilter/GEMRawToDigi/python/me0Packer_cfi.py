import FWCore.ParameterSet.Config as cms

me0packer = cms.EDProducer("ME0DigiToRawModule",
    me0Digi = cms.InputTag("simMuonME0Digis"),
    eventType = cms.int32(0),
    # no DB mapping yet    
    useDBEMap = cms.bool(False),
)
