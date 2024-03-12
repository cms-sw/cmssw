import FWCore.ParameterSet.Config as cms

L1MuTriggerScaleKeysOnlineProd = cms.ESProducer("L1MuTriggerScaleKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('GMT'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),

    # the scales key is identical for both the pt scale
    # and the phi/eta scale objects                                                 
    recordTypes = cms.vstring('L1MuTriggerPtScaleRcd', 'L1MuTriggerScalesRcd'),
    objectTypes = cms.vstring('L1MuTriggerPtScale'   , 'L1MuTriggerScales'   )                                                 
)


# foo bar baz
# ys9fb7kYfmP3S
# YOiQS1NapYxQa
