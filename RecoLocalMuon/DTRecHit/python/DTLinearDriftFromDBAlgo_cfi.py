import FWCore.ParameterSet.Config as cms

# The reconstruction algo and its parameter set
#constant vdrift from DB (and ttrig from DB)
DTLinearDriftFromDBAlgo = cms.PSet(
    recAlgoConfig = cms.PSet(
        # Times outside this window (ns) are considered as coming from previous BXs
        minTime = cms.double(-3.0),
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
        maxTime = cms.double(420.0),
        # The module to be used for ttrig synchronization and its set parameter
        tTrigMode = cms.string('DTTTrigSyncFromDB')
    ),
    recAlgo = cms.string('DTLinearDriftFromDBAlgo')
)

