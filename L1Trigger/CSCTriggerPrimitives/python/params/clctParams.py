import FWCore.ParameterSet.Config as cms

# Parameters for CLCT processors: 2007 and later
clctPhase1 = cms.PSet(
    clctFifoTbins   = cms.uint32(12),
    clctFifoPretrig = cms.uint32(7),
    clctHitPersist  = cms.uint32(4),
    clctDriftDelay  = cms.uint32(2),
    clctNplanesHitPretrig = cms.uint32(3),
    clctNplanesHitPattern = cms.uint32(4),
    clctPidThreshPretrig  = cms.uint32(2),
    clctMinSeparation     = cms.uint32(10),
    # Debug
    verbosity = cms.int32(0),

    # BX to start CLCT finding (poor man's dead-time shortening):
    clctStartBxShift  = cms.int32(0),

    useRun3Patterns = cms.bool(False),
    useComparatorCodes = cms.bool(False),
)

# Parameters for upgrade CLCT processors
clctPhase2 = clctPhase1.clone(
    # decrease possible minimal #HS distance between two CLCTs in a BX from 10 to 5:
    clctMinSeparation     = 5,

    # Turns on algorithms of localized dead-time zones:
    useDeadTimeZoning = cms.bool(True),

    # Width (in #HS) of a fixed dead zone around a key HS:
    clctStateMachineZone = cms.uint32(4),

    # Enables the algo which instead of using the fixed dead zone width,
    # varies it depending on the width of a triggered CLCT pattern
    # (if True, the clctStateMachineZone is ignored):
    useDynamicStateMachineZone = cms.bool(False),

    # Pretrigger HS +- clctPretriggerTriggerZone sets the trigger matching zone
    # which defines how far from pretrigger HS the TMB may look for a trigger HS
    # (it becomes important to do so with localized dead-time zoning):
    # not implemented yet, 2018-10-18, Tao
    clctPretriggerTriggerZone = cms.uint32(224),

    # whether to store the "corrected" CLCT stub time
    # (currently it is median time of all hits in a pattern) into the CSCCLCTDigi bx,
    # and temporary store the regular "key layer hit" time into the CSCCLCTDigi fullBX:
    # not feasible --Tao
    clctUseCorrectedBx = cms.bool(False),
)

clctPhase2GEM = clctPhase2.clone(
    clctNplanesHitPattern = 3
)

clctPSets = cms.PSet(
    clctPhase1 = clctPhase1.clone(),
    clctPhase2 = clctPhase2.clone(),
    clctPhase2GEM = clctPhase2GEM.clone()
)
