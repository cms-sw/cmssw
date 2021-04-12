import FWCore.ParameterSet.Config as cms

simCastorTechTrigDigis = cms.EDProducer("CastorTTRecord",
    CastorDigiCollection = cms.InputTag("simCastorDigis"),
    CastorSignalTS = cms.uint32(4),
    TriggerBitNames = cms.vstring(
        'L1Tech_CASTOR_0.v0',
        'L1Tech_CASTOR_TotalEnergy.v0',
        'L1Tech_CASTOR_EM.v0',
        'L1Tech_CASTOR_HaloMuon.v0'
    ),
    TriggerThresholds = cms.vdouble(
        50, 48000, 1500, 100, 50,
        65000
    ),
    ttpBits = cms.vuint32(60, 61, 62, 63)
)
