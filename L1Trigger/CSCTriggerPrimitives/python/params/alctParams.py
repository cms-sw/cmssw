import FWCore.ParameterSet.Config as cms

# Parameters for ALCT processors: 2007 and later
alctPhase1 = cms.PSet(
    # total number of time bins in the DAQ readout
    alctFifoTbins   = cms.uint32(16),
    alctFifoPretrig = cms.uint32(10),
    # time that is required for the electrons to drift to the
    # anode wires. 15ns drift time --> 45 ns is 3 sigma for the delay
    # this corresponds to 2bx
    alctDriftDelay  = cms.uint32(2),
    # min. number of layers hit for pre-trigger
    alctNplanesHitPretrig = cms.uint32(3),
    # min. number of layers hit for trigger
    alctNplanesHitPattern = cms.uint32(4),
    # min. number of layers hit for pre-trigger
    alctNplanesHitAccelPretrig = cms.uint32(3),
    # min. number of layers hit for trigger
    alctNplanesHitAccelPattern = cms.uint32(4),
    # 0: both collision and accelerator tracks
    # 1: only accelerator tracks
    # 2: only collision tracks
    # 3: prefer accelerator tracks
    alctTrigMode       = cms.uint32(2),
    # preference to collision/accelerator tracks
    alctAccelMode      = cms.uint32(0),
    # L1Accept window width, in 25 ns bins
     alctL1aWindowWidth = cms.uint32(7),
    verbosity = cms.int32(0),

    # Configure early_tbins instead of hardcoding it
    alctEarlyTbins = cms.int32(4),

    # Use narrow pattern mask for ring 1 chambers
    alctNarrowMaskForR1 = cms.bool(False),

    # duration of signal pulse, in 25 ns bins
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
    alctNplanesHitPattern = 4
)

alctPSets = cms.PSet(
    alctPhase1 = alctPhase1.clone(),
    alctPhase2 = alctPhase2.clone(),
    alctPhase2GEM = alctPhase2GEM.clone()
)
