import FWCore.ParameterSet.Config as cms

isotrack_filter = cms.EDFilter("PythiaFilterIsolatedTrack",
                               ModuleLabel     = cms.untracked.InputTag('generator','unsmeared'),
                               MaxSeedEta      = cms.untracked.double(2.3),
                               MinSeedMom      = cms.untracked.double(20.0),
                               MinIsolTrackMom = cms.untracked.double(2.0),
                               IsolCone        = cms.untracked.double(40.0),
                               PixelEfficiency = cms.untracked.double(0.8)
                               )


