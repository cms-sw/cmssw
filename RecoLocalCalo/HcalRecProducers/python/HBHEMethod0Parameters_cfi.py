import FWCore.ParameterSet.Config as cms

# Configuration parameters for Method 0
m0Parameters = cms.PSet(
    firstSampleShift = cms.int32(0),
    samplesToAdd = cms.int32(2),
    correctForPhaseContainment = cms.bool(True),
    correctionPhaseNS = cms.double(6.0),
)
