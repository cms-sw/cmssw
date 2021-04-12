import FWCore.ParameterSet.Config as cms

simBscDigis = cms.EDProducer("BSCTrigger",
    bitNames = cms.vstring(
        'L1Tech_BSC_minBias_inner_threshold1',
        'L1Tech_BSC_minBias_inner_threshold2',
        'L1Tech_BSC_minBias_OR',
        'L1Tech_BSC_HighMultiplicity',
        'L1Tech_BSC_halo_beam2_inner',
        'L1Tech_BSC_halo_beam2_outer',
        'L1Tech_BSC_halo_beam1_inner',
        'L1Tech_BSC_halo_beam1_outer',
        'L1Tech_BSC_minBias_threshold1',
        'L1Tech_BSC_minBias_threshold2',
        'L1Tech_BSC_splash_beam1',
        'L1Tech_BSC_splash_beam2'
    ),
    bitNumbers = cms.vuint32(
        32, 33, 34, 35, 36,
        37, 38, 39, 40, 41,
        42, 43
    ),
    coincidence = cms.double(72.85),
    resolution = cms.double(3.0),
    theHits = cms.InputTag("mix","g4SimHitsBSCHits")
)
