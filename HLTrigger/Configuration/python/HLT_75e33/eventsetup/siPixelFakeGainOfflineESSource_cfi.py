import FWCore.ParameterSet.Config as cms

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
    file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseII/Tilted/EmptyPixelSkimmedGeometry.txt')
)
