import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import *

ootPhotonMatch = photonMatch.clone()
ootPhotonMatch.src = cms.InputTag("reducedOOTPhotons") # RECO objects to match
ootPhotonMatch.matched = cms.InputTag("prunedGenParticles")  # mc-truth particle collection
