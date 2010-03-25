import FWCore.ParameterSet.Config as cms


pvSelector = cms.PSet(
    pvSrc = cms.InputTag('offlinePrimaryVertices'),
    minNdof = cms.double(5.0),
    maxZ = cms.double(15.0)
    )
