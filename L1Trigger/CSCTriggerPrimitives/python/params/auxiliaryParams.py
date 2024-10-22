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

    # Run-3 mode
    run3 = cms.bool(False),

    # comparator-code algorithm to improve
    # CLCT position and bending resolution
    # CCLUT for TMB is NOT planned for startup Run-3
    runCCLUT_TMB = cms.bool(False),
    # CCLUT for OTMB is planned for startup Run-3
    runCCLUT_OTMB = cms.bool(False),

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

auxPSets = cms.PSet(
    commonParam = commonParam.clone(),
    mpcParamRun1 = mpcParamRun1.clone(),
    mpcParamRun2 = mpcParamRun2.clone(),
)
