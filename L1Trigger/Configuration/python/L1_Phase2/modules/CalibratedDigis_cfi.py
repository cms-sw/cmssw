import FWCore.ParameterSet.Config as cms

CalibratedDigis = cms.EDProducer("CalibratedDigis",
    dtDigiTag = cms.InputTag("simMuonDTDigis"),
    flat_calib = cms.int32(0),
    scenario = cms.int32(0),
    tTrigMode = cms.string('DTTTrigSyncFromDB'),
    tTrigModeConfig = cms.PSet(
        debug = cms.untracked.bool(False),
        doT0Correction = cms.bool(True),
        doTOFCorrection = cms.bool(True),
        doWirePropCorrection = cms.bool(True),
        t0Label = cms.string(''),
        tTrigLabel = cms.string(''),
        tofCorrType = cms.int32(0),
        vPropWire = cms.double(24.4),
        wirePropCorrType = cms.int32(0)
    ),
    timeOffset = cms.int32(0)
)
