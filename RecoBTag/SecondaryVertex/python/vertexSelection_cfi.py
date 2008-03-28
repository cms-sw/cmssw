import FWCore.ParameterSet.Config as cms

vertexSelection = cms.PSet(
    sortCriterium = cms.string('dist3dError')
)

