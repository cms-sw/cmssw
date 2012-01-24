import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonSeedGenerator.ancientMuonSeed_cfi import *

L2OfflineMuonSeeds = ancientMuonSeed.clone(
    CSCRecSegmentLabel = cms.InputTag("hltCscSegments"),
    DTRecSegmentLabel = cms.InputTag("hltDt4DSegments")
    )

