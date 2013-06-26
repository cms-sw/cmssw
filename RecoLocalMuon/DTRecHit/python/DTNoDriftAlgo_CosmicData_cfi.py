import FWCore.ParameterSet.Config as cms

# The reconstruction algo and its parameter set
DTNoDriftAlgo_CosmicData = cms.PSet(
    recAlgoConfig = cms.PSet(
        # DUMMY
        #
        # Fixed Drift distance (cm)                
        fixedDrift = cms.double(1.0),
        #
        # DUMMY
        # Dummy ttrig parameters required by DTRecHitBaseAlgo.cc
        tTrigMode = cms.string('DTTTrigSyncFromDB'),
        # Times outside this window (ns) are considered as coming from previous BXs
        minTime = cms.double(1000.0),
        # Cell resolution (cm)
        hitResolution = cms.double(1.0),
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
            doT0Correction = cms.bool(False),
            debug = cms.untracked.bool(False),
            tTrigLabel = cms.string('cosmics')
        ),
        maxTime = cms.double(3500.0)
    ),
    recAlgo = cms.string('DTNoDriftAlgo')
)

