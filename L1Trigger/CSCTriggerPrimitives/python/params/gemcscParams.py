import FWCore.ParameterSet.Config as cms

# GEM coincidence pad processors
copadParamGE11 = cms.PSet(
    verbosity = cms.uint32(0),
    maxDeltaPad = cms.uint32(8),
    maxDeltaRoll = cms.uint32(1),
    maxDeltaBX = cms.uint32(0)
)

copadParamGE21 = copadParamGE11.clone()

gemcscPSets = cms.PSet(
    copadParamGE11 = copadParamGE11.clone(),
    copadParamGE21 = copadParamGE21.clone(),
)
