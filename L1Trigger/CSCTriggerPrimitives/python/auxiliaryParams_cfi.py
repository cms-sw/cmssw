import FWCore.ParameterSet.Config as cms

# Parameters common for all boards
commonParam = cms.PSet(
    # Master flag for SLHC studies
    isSLHC = cms.bool(False),

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

    # offset between the ALCT and CLCT central BX in simulation
    alctClctOffset = cms.uint32(1),

    runME11Up = cms.bool(False),
    runME21Up = cms.bool(False),
    runME31Up = cms.bool(False),
    runME41Up = cms.bool(False),

    runME11ILT = cms.bool(False),
    runME21ILT = cms.bool(False),
    useClusters = cms.bool(False),
)

# MPC sorter config for Run1
mpcParamRun1 = cms.PSet(
    sortStubs = cms.bool(True),
    dropInvalidStubs = cms.bool(True),
    dropLowQualityStubs = cms.bool(True),
)

# MPC sorter config for Run2 and beyond
mpcParamRun2 = cms.PSet(
    sortStubs = cms.bool(False),
    dropInvalidStubs = cms.bool(False),
    dropLowQualityStubs = cms.bool(False),
)

# GEM coincidence pad processors
copadParamGE11 = cms.PSet(
    verbosity = cms.uint32(0),
    maxDeltaPad = cms.uint32(2),
    maxDeltaRoll = cms.uint32(1),
    maxDeltaBX = cms.uint32(0)
)

copadParamGE21 = copadParamGE11.clone()
