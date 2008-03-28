import FWCore.ParameterSet.Config as cms

ecaldigitorawzerosup = cms.EDFilter("EcalDigiToRaw",
    InstanceEB = cms.string('SRPebDigis'),
    InstanceEE = cms.string('SRPeeDigis'),
    DoEndCap = cms.untracked.bool(True),
    labelTT = cms.InputTag("ecalTriggerPrimitiveDigis"),
    Label = cms.string('ecalDigis'),
    debug = cms.untracked.bool(False),
    labelEESRFlags = cms.InputTag("ecalDigis","eeSrFlags"),
    WriteSRFlags = cms.untracked.bool(True),
    WriteTowerBlock = cms.untracked.bool(True),
    labelEBSRFlags = cms.InputTag("ecalDigis","ebSrFlags"),
    WriteTCCBlock = cms.untracked.bool(True),
    DoBarrel = cms.untracked.bool(True)
)


