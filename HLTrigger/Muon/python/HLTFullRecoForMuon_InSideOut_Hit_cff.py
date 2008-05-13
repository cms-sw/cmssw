import FWCore.ParameterSet.Config as cms

import RecoMuon.L2MuonSeedGenerator.L2MuonSeeds_cfi
# L2 seeds from L1 input
hltL2MuonSeeds = RecoMuon.L2MuonSeedGenerator.L2MuonSeeds_cfi.L2MuonSeeds.clone()
import RecoMuon.TrackerSeedGenerator.TSGFromL2_cfi
hltL3TrajectorySeed = RecoMuon.TrackerSeedGenerator.TSGFromL2_cfi.hltL3TrajectorySeedFromL2.clone()
import RecoTracker.CkfPattern.CkfTrajectories_cfi
hltL3TrackCandidateFromL2 = RecoTracker.CkfPattern.CkfTrajectories_cfi.ckfTrajectories.clone()
# L3 regional reconstruction
from RecoMuon.L3MuonProducer.L3Muons_cff import *
import RecoMuon.L3MuonProducer.L3Muons_cfi
hltL3Muons = RecoMuon.L3MuonProducer.L3Muons_cfi.L3Muons.clone()
# Pixel tracking for muon isolation
from HLTrigger.Configuration.common.Vertexing_cff import *
hltL3MuonTracks = cms.Sequence(hltL3TrajectorySeed*hltL3TrackCandidateFromL2*hltL3Muons)
pixelTracksForMuons = cms.Sequence(pixelTracks)
hltL2MuonSeeds.GMTReadoutCollection = 'gtDigis'
hltL3TrajectorySeed.TSGFromCombinedHits.PSetNames = ['firstTSG', 'secondTSG']
hltL3TrackCandidateFromL2.SeedProducer = 'hltL3TrajectorySeed'
hltL3TrackCandidateFromL2.TrajectoryBuilder = 'muonCkfTrajectoryBuilder'
hltL3TrackCandidateFromL2.trackCandidateAlso = True
hltL3Muons.MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx")
hltL3Muons.L3TrajBuilderParameters.l3SeedLabel = 'donotgetSEED'
hltL3Muons.L3TrajBuilderParameters.tkTrajLabel = 'hltL3TrackCandidateFromL2'

