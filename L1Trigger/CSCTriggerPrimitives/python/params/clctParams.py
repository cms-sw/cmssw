import FWCore.ParameterSet.Config as cms

# Parameters for CLCT processors: 2007 and later
clctPhase1 = cms.PSet(
    # total number of time bins in the DAQ readout
    clctFifoTbins   = cms.uint32(12),
    # start time of cathode raw hits in DAQ readout
    clctFifoPretrig = cms.uint32(7),
    # duration of signal pulse, in 25 ns bins
    clctHitPersist  = cms.uint32(4),
    # time that is required for the electrons to drift to the
    # cathode strips. 15ns drift time --> 45 ns is 3 sigma for the delay
    # this corresponds to 2bx
    clctDriftDelay  = cms.uint32(2),
    # min. number of layers hit for pre-trigger
    clctNplanesHitPretrig = cms.uint32(3),
    # min. number of layers hit for trigger
    clctNplanesHitPattern = cms.uint32(4),
    # lower threshold on pattern id
    clctPidThreshPretrig  = cms.uint32(2),
    # region of busy key strips
    clctMinSeparation     = cms.uint32(10),

    # Turns on algorithms of localized dead-time zones:
    useDeadTimeZoning = cms.bool(False),

    # Debug
    verbosity = cms.int32(0),

    # BX to start CLCT finding (poor man's dead-time shortening):
    clctStartBxShift  = cms.int32(0),
)

# Parameters for upgrade CLCT processors
clctPhase2 = clctPhase1.clone(
    # decrease possible minimal #HS distance between two CLCTs in a BX from 10 to 5:
    clctMinSeparation     = 5,

    # Turns on algorithms of localized dead-time zones:
    useDeadTimeZoning = True,

    # Width (in #HS) of a fixed dead zone around a key HS:
    clctStateMachineZone = cms.uint32(4),

    # Pretrigger HS +- clctPretriggerTriggerZone sets the trigger matching zone
    # which defines how far from pretrigger HS the TMB may look for a trigger HS
    # (it becomes important to do so with localized dead-time zoning):
    # not implemented yet, 2018-10-18, Tao Huang
    clctPretriggerTriggerZone = cms.uint32(224),

    # whether to store the "corrected" CLCT stub time
    # (currently it is median time of all hits in a pattern) into the CSCCLCTDigi bx,
    # and temporary store the regular "key layer hit" time into the CSCCLCTDigi fullBX:
    # not feasible in firmware -- Tao Huang
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
