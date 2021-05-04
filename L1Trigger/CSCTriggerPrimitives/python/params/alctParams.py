import FWCore.ParameterSet.Config as cms

# Parameters for ALCT processors: 2007 and later
alctPhase1 = cms.PSet(
    alctFifoTbins   = cms.uint32(16),
    alctFifoPretrig = cms.uint32(10),
    alctDriftDelay  = cms.uint32(2),
    alctNplanesHitPretrig = cms.uint32(3),
    alctNplanesHitPattern = cms.uint32(4),
    alctNplanesHitAccelPretrig = cms.uint32(3),
    alctNplanesHitAccelPattern = cms.uint32(4),
    alctTrigMode       = cms.uint32(2),
    alctAccelMode      = cms.uint32(0),
    alctL1aWindowWidth = cms.uint32(7),
    verbosity = cms.int32(0),

    # Configure early_tbins instead of hardcoding it
    alctEarlyTbins = cms.int32(4),

    # Use narrow pattern mask for ring 1 chambers
    alctNarrowMaskForR1 = cms.bool(False),

    # configured, not hardcoded, hit persistency
    alctHitPersist  = cms.uint32(6),

    # configure, not hardcode, up to how many BXs in the past
    # ghost cancellation in neighboring WGs may happen
    alctGhostCancellationBxDepth = cms.int32(4),

    # whether to compare the quality of stubs in neighboring WGs in the past
    # to the quality of a stub in current WG
    # when doing ghost cancellation
    alctGhostCancellationSideQuality = cms.bool(False),

    # how soon after pretrigger and alctDriftDelay can next pretrigger happen?
    alctPretrigDeadtime = cms.uint32(4),

    # whether to store the "corrected" ALCT stub time
    # (currently it is median time of particular hits in a pattern) into the CSCCLCTDigi bx,
    # and temporary store the regular "key layer hit" time into the CSCCLCTDigi fullBX:
    alctUseCorrectedBx = cms.bool(False)
)

# Parameters for upgrade ALCT processors
alctPhase2 = alctPhase1.clone(
    alctNarrowMaskForR1 = True,
    alctGhostCancellationBxDepth = 1,
    alctGhostCancellationSideQuality = True,
    alctPretrigDeadtime = 0,
    alctUseCorrectedBx = True,
)

alctPhase2GEM = alctPhase2.clone(
    alctNplanesHitPattern = 3
)

alctPSets = cms.PSet(
    alctPhase1 = alctPhase1.clone(),
    alctPhase2 = alctPhase2.clone(),
    alctPhase2GEM = alctPhase2GEM.clone()
)
