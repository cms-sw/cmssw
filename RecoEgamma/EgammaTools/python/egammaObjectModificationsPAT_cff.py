import FWCore.ParameterSet.Config as cms

egamma_modifications = cms.VPSet(
    cms.PSet( modifierName = cms.string('EGFull5x5ShowerShapeModifierFromValueMaps') )
)
