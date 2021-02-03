import FWCore.ParameterSet.Config as cms

ecalNextToDeadChannelESProducer = cms.ESProducer("EcalNextToDeadChannelESProducer",
    channelStatusThresholdForDead = cms.int32(12)
)
