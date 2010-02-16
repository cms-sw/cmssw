import FWCore.ParameterSet.Config as cms

# The reconstruction algo and its parameter set
#constant vdrift (and ttrig from DB)
DTLinearDriftAlgo_CosmicData = cms.PSet(
    recAlgoConfig = cms.PSet(
        # The module to be used for ttrig synchronization and its set parameter
        tTrigMode = cms.string('DTTTrigSyncFromDB'),
        #         # Drift velocity for MB1 W1 (unfluxed chamber during MTCC)
        # double driftVelocityMB1W1 = 0.00570
        # Times outside this window (ns) are considered as coming from previous BXs
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
            doTOFCorrection = cms.bool(False),
            tofCorrType = cms.int32(0),
            wirePropCorrType = cms.int32(0),
            # Switch on/off the correction for the signal propagation along the wire
            doWirePropCorrection = cms.bool(False),
            # Switch on/off the T0 correction from pulses
            doT0Correction = cms.bool(True),
            debug = cms.untracked.bool(False),
            tTrigLabel = cms.string('cosmics')
        ),
        maxTime = cms.double(420.0)
    ),
    recAlgo = cms.string('DTLinearDriftAlgo')
)

