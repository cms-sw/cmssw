import FWCore.ParameterSet.Config as cms

totemRPClusterProducer = cms.EDProducer("TotemRPClusterProducer",
    verbosity = cms.int32(0),
    tagDigi = cms.InputTag("totemRPRawToDigi", "RP")
)
