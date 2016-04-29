import FWCore.ParameterSet.Config as cms

# Make one TrackCand for each seeder
import FastSimulation.Tracking.TrackCandidateProducer_cfi
hltL3TrackCandidateFromL2OIState = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("hltL3TrajSeedOIState"),
    SplitHits = cms.bool(False),
    maxSeedMatchEstimator = cms.untracked.double(200)
    )
hltL3TrackCandidateFromL2OIHit = hltL3TrackCandidateFromL2OIState.clone()
hltL3TrackCandidateFromL2OIHit.src = "hltL3TrajSeedOIHit"    
hltL3TrackCandidateFromL2IOHit = hltL3TrackCandidateFromL2OIState.clone()
hltL3TrackCandidateFromL2IOHit.src = "hltL3TrajSeedIOHit"

# CKFTrackCandidateMaker
hltMuCkfTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltMuCkfTrackCandidates.src = cms.InputTag("hltMuTrackSeeds")
hltMuCkfTrackCandidates.SplitHits = False

# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
hltMuCtfTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltMuCtfTracks.src = 'hltMuCkfTrackCandidates'
hltMuCtfTracks.TTRHBuilder = 'WithoutRefit'
