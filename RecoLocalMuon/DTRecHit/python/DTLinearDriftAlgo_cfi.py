import FWCore.ParameterSet.Config as cms

DTLinearDriftAlgo = cms.PSet(
    recAlgoConfig = cms.PSet(
        # The module to be used for ttrig synchronization and its parameter set
        tTrigMode = cms.string('DTTTrigSyncFromDB'),
        # Times outside this window (ns) are considered 
        # as coming from previous BXs
        minTime = cms.double(-3.0),
        # Drift velocity (cm/ns)                
        driftVelocity = cms.double(0.00543),
        # Cell resolution (cm)
        hitResolution = cms.double(0.02),
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
        maxTime = cms.double(415.0)
    ),
    recAlgo = cms.string('DTLinearDriftAlgo')
)

