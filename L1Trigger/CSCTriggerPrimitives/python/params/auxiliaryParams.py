import FWCore.ParameterSet.Config as cms

# Parameters common for all boards
commonParam = cms.PSet(
    # Master flag for Phase-2 studies
    runPhase2 = cms.bool(False),

    # Debug
    verbosity = cms.int32(0),

    ## During Run-1, ME1a strips were triple-ganged
    ## Effectively, this means there were only 16 strips
    ## As of Run-2, ME1a strips are unganged,
    ## which increased the number of strips to 48
    gangedME1a = cms.bool(True),

    # flags to optionally disable finding stubs in ME42 or ME1a
    disableME1a = cms.bool(False),
    disableME42 = cms.bool(False),

    # switch on HL-LHC algorithms
    runME11Up = cms.bool(False),
    runME21Up = cms.bool(False),
    runME31Up = cms.bool(False),
    runME41Up = cms.bool(False),

    # GEM-CSC integrated triggers
    runME11ILT = cms.bool(False),
    runME21ILT = cms.bool(False),

    # comparator-code algorithm to improve
    # CLCT position and bending resolution
    runCCLUT = cms.bool(False),

    ## Phase-2 version is not needed for Run-3
    enableAlctPhase2 = cms.bool(False)
)

# MPC sorter config for Run1
mpcParamRun1 = cms.PSet(
    sortStubs = cms.bool(True),
    dropInvalidStubs = cms.bool(True),
    dropLowQualityStubs = cms.bool(True),
    maxStubs = cms.uint32(3),
)

# MPC sorter config for Run2 and beyond
mpcParamRun2 = mpcParamRun1.clone(
    sortStubs = False,
    dropInvalidStubs = False,
    dropLowQualityStubs = False,
    maxStubs = 18,
)

# GEM coincidence pad processors
copadParamGE11 = cms.PSet(
    verbosity = cms.uint32(0),
    maxDeltaPad = cms.uint32(4),
    maxDeltaRoll = cms.uint32(1),
    maxDeltaBX = cms.uint32(0)
)

copadParamGE21 = copadParamGE11.clone()

auxPSets = cms.PSet(
    commonParam = commonParam.clone(),
    mpcParamRun1 = mpcParamRun1.clone(),
    mpcParamRun2 = mpcParamRun2.clone(),
    copadParamGE11 = copadParamGE11.clone(),
    copadParamGE21 = copadParamGE21.clone()
)
