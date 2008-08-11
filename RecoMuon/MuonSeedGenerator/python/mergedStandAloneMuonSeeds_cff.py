import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cfi.py import MuonSeed



mergedStandAloneMuonSeeds = cms.EDProducer("MuonSeedMerger",
                                           SeedCollections = cms.VInputTag(cms.InputTag())
    )
