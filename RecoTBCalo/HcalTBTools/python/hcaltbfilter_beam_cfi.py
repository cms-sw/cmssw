import FWCore.ParameterSet.Config as cms

hcaltbfilter_beam = cms.EDFilter("HcalTBTriggerFilter",
    AllowLED = cms.bool(False),
    AllowPedestalOutSpill = cms.bool(False),
    AllowLaser = cms.bool(False),
    AllowPedestal = cms.bool(False),
    AllowBeam = cms.bool(True),
    AllowPedestalInSpill = cms.bool(False),
    hcalTBTriggerDataTag = cms.InputTag("tbunpack")
)


