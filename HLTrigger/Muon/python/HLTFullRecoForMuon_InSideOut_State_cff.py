import FWCore.ParameterSet.Config as cms

import copy
from RecoMuon.L2MuonSeedGenerator.L2MuonSeeds_cfi import *
# L2 seeds from L1 input
hltL2MuonSeeds = copy.deepcopy(L2MuonSeeds)
import copy
from RecoMuon.TrackerSeedGenerator.TSGFromL2_cfi import *
hltL3TrajectorySeed = copy.deepcopy(hltL3TrajectorySeedFromL2)
import copy
from RecoTracker.CkfPattern.CkfTrajectories_cfi import *
hltL3TrackCandidateFromL2 = copy.deepcopy(ckfTrajectories)
# L3 regional reconstruction
from RecoMuon.L3MuonProducer.L3Muons_cff import *
import copy
from RecoMuon.L3MuonProducer.L3Muons_cfi import *
hltL3Muons = copy.deepcopy(L3Muons)
# Pixel tracking for muon isolation
from HLTrigger.Configuration.common.Vertexing_cff import *
hltL3MuonTracks = cms.Sequence(hltL3TrajectorySeed*hltL3TrackCandidateFromL2*hltL3Muons)
pixelTracksForMuons = cms.Sequence(pixelTracks)
hltL2MuonSeeds.GMTReadoutCollection = 'gtDigis'
hltL3TrajectorySeed.tkSeedGenerator = 'TSGForRoadSearchIOpxl'
hltL3TrajectorySeed.MuonTrackingRegionBuilder = cms.PSet()
hltL3TrackCandidateFromL2.SeedProducer = 'hltL3TrajectorySeed'
hltL3TrackCandidateFromL2.TrajectoryBuilder = 'muonCkfTrajectoryBuilder'
hltL3TrackCandidateFromL2.trackCandidateAlso = True
MuonCkfTrajectoryBuilder.useSeedLayer = True
hltL3Muons.MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx")
hltL3Muons.L3TrajBuilderParameters.l3SeedLabel = 'donotgetSEED'
hltL3Muons.L3TrajBuilderParameters.tkTrajLabel = 'hltL3TrackCandidateFromL2'

