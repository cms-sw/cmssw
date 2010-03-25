import FWCore.ParameterSet.Config as cms

# the PSET should be swapped out eventually and replaced with this import statement
# for now, this is ok...
# from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import HcalNoiseParameterSet

HcalNoiseParameterSet = cms.PSet(
    # define hit energy thesholds
    minRecHitE = cms.double(1.5),
    minLowHitE = cms.double(10.0),
    minHighHitE = cms.double(25.0),
    
    # define energy threshold for "problematic" cuts
    pMinERatio = cms.double(25.0),
    pMinEZeros = cms.double(5.0),
    pMinEEMF = cms.double(10.0),
    
    # define energy threshold for loose/tight/high level cuts
    minERatio = cms.double(50.0),
    minEZeros = cms.double(10.0),
    minEEMF = cms.double(20.0),
    
    # define problematic RBX
    pMinE = cms.double(100.0),
    pMinRatio = cms.double(0.75),
    pMaxRatio = cms.double(0.90),
    pMinHPDHits = cms.int32(10),
    pMinRBXHits = cms.int32(20),
    pMinHPDNoOtherHits = cms.int32(7),
    pMinZeros = cms.int32(4),
    pMinLowEHitTime = cms.double(-6.0),
    pMaxLowEHitTime = cms.double(6.0),
    pMinHighEHitTime = cms.double(-4.0),
    pMaxHighEHitTime = cms.double(5.0),
    pMaxHPDEMF = cms.double(0.02),
    pMaxRBXEMF = cms.double(0.02),

    # define loose noise cuts
    lMinRatio = cms.double(0.65),
    lMaxRatio = cms.double(0.95),
    lMinHPDHits = cms.int32(17),
    lMinRBXHits = cms.int32(999),
    lMinHPDNoOtherHits = cms.int32(10),
    lMinZeros = cms.int32(10),
    lMinLowEHitTime = cms.double(-9999.0),
    lMaxLowEHitTime = cms.double(9999.0),
    lMinHighEHitTime = cms.double(-7.0),
    lMaxHighEHitTime = cms.double(6.0),

    # define tight noise cuts
    tMinRatio = cms.double(0.73),
    tMaxRatio = cms.double(0.85),
    tMinHPDHits = cms.int32(16),
    tMinRBXHits = cms.int32(50),
    tMinHPDNoOtherHits = cms.int32(9),
    tMinZeros = cms.int32(8),
    tMinLowEHitTime = cms.double(-9999.0),
    tMaxLowEHitTime = cms.double(9999.0),
    tMinHighEHitTime = cms.double(-5.0),
    tMaxHighEHitTime = cms.double(4.0),

    # define high level noise cuts
    hlMaxHPDEMF = cms.double(-999.),
    hlMaxRBXEMF = cms.double(0.01)
    )


hltHcalMETNoiseFilter = cms.EDFilter(
    "HLTHcalMETNoiseFilter",

    # noise parameters needed for RBX noise algorithm
    HcalNoiseParameterSet,

    # collections to get
    HcalNoiseRBXCollection = cms.InputTag("hcalnoise"),
    
    # set to 0 if you want to accept all events
    severity = cms.int32(1),

    # consider the top N=numRBXsToConsider RBXs by energy in the event
    numRBXsToConsider = cms.int32(4),

    # require coincidence between the High-Level (EMF) filter and the other filters
    needHighLevelCoincidence = cms.bool(True),

    # filters to use
    useLooseRatioFilter = cms.bool(True),
    useLooseHitsFilter = cms.bool(True),
    useLooseZerosFilter = cms.bool(True),
    useLooseTimingFilter = cms.bool(False),
    useTightRatioFilter = cms.bool(False),
    useTightHitsFilter = cms.bool(False),
    useTightZerosFilter = cms.bool(False),
    useTightTimingFilter = cms.bool(False),

    )

