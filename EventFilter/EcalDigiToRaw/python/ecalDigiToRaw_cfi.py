import FWCore.ParameterSet.Config as cms

ecaldigitorawzerosup = cms.EDFilter("EcalDigiToRaw",
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
    WriteTCCBlock = cms.untracked.bool(True),
    DoBarrel = cms.untracked.bool(True)
)



