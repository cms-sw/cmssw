import FWCore.ParameterSet.Config as cms

# TODO: clean this up, move to FastSimulation/Muons

# L3 regional reconstruction
from FastSimulation.Muons.L3Muons_cff import *

import FastSimulation.Muons.TSGFromL2_cfi as TSG
from FastSimulation.Muons.TSGFromL2_cfi import OIHitPropagators as OIHProp
hltL3TrajSeedOIHit = TSG.l3seeds("OIHitCascade")
hltL3TrajSeedOIHit.ServiceParameters.Propagators = cms.untracked.vstring()
OIHProp(hltL3TrajSeedOIHit,hltL3TrajSeedOIHit.TkSeedGenerator.iterativeTSG)
hltL3TrajSeedIOHit = TSG.l3seeds("IOHitCascade")
hltL3NoFiltersTrajSeedOIHit = TSG.l3seeds("OIHitCascade")
hltL3NoFiltersTrajSeedOIHit.ServiceParameters.Propagators = cms.untracked.vstring()
OIHProp(hltL3NoFiltersTrajSeedOIHit,hltL3NoFiltersTrajSeedOIHit.TkSeedGenerator.iterativeTSG)
hltL3NoFiltersTrajSeedIOHit = TSG.l3seeds("IOHitCascade")

## Make one TrackCand for each seeder
from FastSimulation.Muons.TrackCandidateFromL2_cfi import *
hltL3TrackCandidateFromL2OIState = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2OIState.src = "hltL3TrajSeedOIState"
hltL3TrackCandidateFromL2OIHit = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2OIHit.src = "hltL3TrajSeedOIHit"    
hltL3TrackCandidateFromL2IOHit = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2IOHit.src = "hltL3TrajSeedIOHit"
hltL3TrackCandidateFromL2NoVtx = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2NoVtx.src = "hltL3TrajectorySeedNoVtx"


# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltMuCkfTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltMuCkfTrackCandidates.src = cms.InputTag("hltMuTrackSeeds")
hltMuCkfTrackCandidates.SplitHits = False

# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltMuCtfTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltMuCtfTracks.src = 'hltMuCkfTrackCandidates'
hltMuCtfTracks.TTRHBuilder = 'WithoutRefit'
hltMuCtfTracks.Fitter = 'KFFittingSmoother'
hltMuCtfTracks.Propagator = 'PropagatorWithMaterial'
