import FWCore.ParameterSet.Config as cms

HltZeePMDocaFilter = cms.EDFilter("HLTPMDocaFilter",
    docaDiffPerpCutHigh = cms.double(0.055691), ## three sigma
    candTag = cms.InputTag("HltZeePMMassFilter"),
    docaDiffPerpCutLow = cms.double(0.0),
    nZcandcut = cms.int32(1), ## by candidate we now mean Z candidate
    saveTags = cms.bool( False )
)


