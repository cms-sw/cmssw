# The following comments couldn't be translated into the new config version:

#EB-
#EB+
#EE+
import FWCore.ParameterSet.Config as cms

ecaldigitorawzerosup = cms.EDProducer("EcalDigiToRaw",
    InstanceEB = cms.string('SRPebDigis'),
    InstanceEE = cms.string('SRPeeDigis'),
    DoEndCap = cms.untracked.bool(True),
    labelTT = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    Label = cms.string('simEcalDigis'),
    debug = cms.untracked.bool(False),
    labelEESRFlags = cms.InputTag("simEcalDigis","eeSrFlags"),
    WriteSRFlags = cms.untracked.bool(True),
    WriteTowerBlock = cms.untracked.bool(True),
    labelEBSRFlags = cms.InputTag("simEcalDigis","ebSrFlags"),
    listDCCId = cms.untracked.vint32(1, 2, 3, 4, 5,
        6, 7, 8, 9,

        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
        25, 26, 27,

        28, 29, 30, 
        31, 32, 33, 34, 35, 
        36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 

        46, 47, 48, 49, 50, 
        51, 52, 53, 54),
    WriteTCCBlock = cms.untracked.bool(True),
    DoBarrel = cms.untracked.bool(True)
)



