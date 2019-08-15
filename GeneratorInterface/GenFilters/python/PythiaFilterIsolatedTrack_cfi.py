import FWCore.ParameterSet.Config as cms

isotrack_filter = cms.EDFilter("PythiaFilterIsolatedTrack",
                               moduleLabel     = cms.untracked.InputTag('generator','unsmeared'),
                               maxSeedEta      = cms.untracked.double(2.3),
                               minSeedEta      = cms.untracked.double(0.0),
                               minSeedMom      = cms.untracked.double(20.0),
                               minIsolTrackMom = cms.untracked.double(2.0),
                               isolCone        = cms.untracked.double(40.0),
                               onlyHadrons     = cms.untracked.bool(True)
                               )


