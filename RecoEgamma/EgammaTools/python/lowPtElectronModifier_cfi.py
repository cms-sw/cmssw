import FWCore.ParameterSet.Config as cms

lowPtElectronModifier = cms.PSet(
    modifierName = cms.string('LowPtElectronModifier'),
    beamSpot = cms.InputTag('offlineBeamSpot'),
    conversions = cms.InputTag('gsfTracksOpenConversions:gsfTracksOpenConversions'),
    addExtraUserVars = cms.bool(True),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
)
