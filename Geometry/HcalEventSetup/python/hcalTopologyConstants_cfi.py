import FWCore.ParameterSet.Config as cms

hcalTopologyConstants = cms.PSet(
    mode = cms.string('HcalTopologyMode::LHC'),
    maxDepthHB = cms.int32(2),
    maxDepthHE = cms.int32(3)
)
