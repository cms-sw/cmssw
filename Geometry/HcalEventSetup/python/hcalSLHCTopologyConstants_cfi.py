import FWCore.ParameterSet.Config as cms

hcalTopologyConstants = cms.PSet(
    mode = cms.string('HcalTopologyMode::SLHC'),
    maxDepthHB = cms.int32(4),
    maxDepthHE = cms.int32(5)
)
