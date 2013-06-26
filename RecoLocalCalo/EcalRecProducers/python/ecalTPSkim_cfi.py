import FWCore.ParameterSet.Config as cms

# rechit producer
ecalTPSkim = cms.EDProducer("EcalTPSkimmer",

    # channel flags for which we want to keep the TP
    chStatusToSelectTP = cms.vuint32( 13 ),

    # whether to execute the module at all
    skipModule = cms.bool(False),

    # keep TP for barrel/endcap?
    doBarrel  = cms.bool(True),
    doEndcap  = cms.bool(True),

    tpInputCollection = cms.InputTag("ecalDigis:EcalTriggerPrimitives"),
    tpOutputCollection = cms.string("")
)
