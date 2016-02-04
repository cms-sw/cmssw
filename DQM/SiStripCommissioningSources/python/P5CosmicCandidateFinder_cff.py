import FWCore.ParameterSet.Config as cms

from RecoTracker.SingleTrackPattern.CosmicCandidateFinder_cfi import *
cosmicCandidateFinder.pixelRecHits = cms.InputTag('dummy','dummy')
cosmicCandidateFinder.MinHits = cms.int32(4)
cosmicCandidateFinder.Chi2Cut = cms.double(300.0)
cosmicCandidateFinder.GeometricStructure = cms.untracked.string('STANDARD')
