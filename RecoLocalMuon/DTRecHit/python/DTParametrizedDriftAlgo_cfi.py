import FWCore.ParameterSet.Config as cms

# The reconstruction algo and its parameter set
DTParametrizedDriftAlgo = cms.PSet(
    recAlgoConfig = cms.PSet(
        # The module to be used for ttrig synchronization and its parameter set
        tTrigMode = cms.string('DTTTrigSyncFromDB'),
        minTime = cms.double(-3.0),
        interpolate = cms.bool(True),
        debug = cms.untracked.bool(False),
        tTrigModeConfig = cms.PSet(
            # The velocity of signal propagation along the wire (cm/ns)
            vPropWire = cms.double(24.4),
            # Switch on/off the TOF correction for particles
            doTOFCorrection = cms.bool(True),
            tofCorrType = cms.int32(1),
            # The ttrig from the time box fit is defined as mean + kFactor * sigma
            kFactor = cms.double(-2.0),
            wirePropCorrType = cms.int32(1),
            # Switch on/off the correction for the signal propagation along the wire
            doWirePropCorrection = cms.bool(True),
            # Switch on/off the TOF correction from pulses
            doT0Correction = cms.bool(True),
            debug = cms.untracked.bool(False)
        ),
        maxTime = cms.double(415.0)
    ),
    recAlgo = cms.string('DTParametrizedDriftAlgo')
)

