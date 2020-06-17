import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonSeedGenerator.ancientMuonSeed_cfi import *

L2OfflineMuonSeeds = ancientMuonSeed.clone(
    CSCRecSegmentLabel = 'hltCscSegments',
    DTRecSegmentLabel  = 'hltDt4DSegments'
)
