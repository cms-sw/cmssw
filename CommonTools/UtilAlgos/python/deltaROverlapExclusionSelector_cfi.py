import FWCore.ParameterSet.Config as cms

deltaROverlapExclusionSelector = cms.EDFilter( "DeltaROverlapExclusionSelector",
   src = cms.InputTag(""),
   overlap = cms.InputTag(""),
   maxDeltaR = cms.double(0.01),
)

