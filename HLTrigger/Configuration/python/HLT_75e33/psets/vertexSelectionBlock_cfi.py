import FWCore.ParameterSet.Config as cms

vertexSelectionBlock = cms.PSet(
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)