import FWCore.ParameterSet.Config as cms


pvSelector = cms.PSet(
    NPV     = cms.int32(1),
    pvSrc = cms.InputTag('offlinePrimaryVertices'),
    minNdof = cms.double(4.0),
    maxZ = cms.double(15.0),
    maxRho = cms.double(2.0)
    )
