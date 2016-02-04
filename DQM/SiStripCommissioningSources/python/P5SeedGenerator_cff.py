import FWCore.ParameterSet.Config as cms

from RecoTracker.SpecialSeedGenerators.CosmicSeed_cfi import cosmicseedfinder
cosmicseedfinder.SeedPt = cms.double(1.0)
