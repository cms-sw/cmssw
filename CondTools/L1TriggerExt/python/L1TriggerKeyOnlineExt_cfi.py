import FWCore.ParameterSet.Config as cms

L1TriggerKeyOnlineExt = cms.ESProducer("L1TriggerKeyOnlineProdExt",
    subsystemLabels = cms.vstring( 'uGT', 'uGTrs', 'uGMT', 'CALO', 'BMTF', 'OMTF', 'EMTF' )
)


