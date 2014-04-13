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
            tofCorrType = cms.int32(0),
            wirePropCorrType = cms.int32(0),
            # Switch on/off the correction for the signal propagation along the wire
            doWirePropCorrection = cms.bool(True),
            # Switch on/off the TOF correction from pulses
            doT0Correction = cms.bool(True),
            debug = cms.untracked.bool(False),
            tTrigLabel = cms.string('')
        ),
        maxTime = cms.double(420.0),
        # Forcing Step 2 to go back to digi time 
        stepTwoFromDigi = cms.bool(False),
        # The module to be used for ttrig synchronization and its set parameter
        tTrigMode = cms.string('DTTTrigSyncFromDB'),
        # perform a correction to vdrift in MB1s of external wheels
        doVdriftCorr = cms.bool(True),
        useUncertDB = cms.bool(True)
    ),
    recAlgo = cms.string('DTLinearDriftFromDBAlgo')
)

