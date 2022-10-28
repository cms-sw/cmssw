import FWCore.ParameterSet.Config as cms

# The reconstruction algo and its parameter set
# constant vdrift from DB (and ttrig from DB)

CalibratedDigis = cms.EDProducer("CalibratedDigis",
                                 tTrigModeConfig = cms.PSet(
        # The velocity of signal propagation along the wire (cm/ns)
        vPropWire = cms.double(24.4),
        # Switch on/off the TOF correction for particles
        doTOFCorrection = cms.bool(True),
        tofCorrType = cms.int32(0),
        wirePropCorrType = cms.int32(0),
        # Switch on/off the correction for the signal propagation along the wire
        doWirePropCorrection = cms.bool(True),
        # Switch on/off the TOF correction from pulses
        doT0Correction = cms.bool(True),
        debug = cms.untracked.bool(False),
        tTrigLabel = cms.string(''),
        t0Label = cms.string('')
        ),
                                 tTrigMode = cms.string('DTTTrigSyncFromDB'),
                                 timeOffset = cms.int32(0),
                                 flat_calib = cms.int32(0),
                                 scenario = cms.int32(0),
                                 dtDigiTag = cms.InputTag("muonDTDigis")
                                 )




