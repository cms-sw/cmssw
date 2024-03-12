import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import *

ootPhotonMatch = photonMatch.clone()
ootPhotonMatch.src = cms.InputTag("ootPhotons") # RECO objects to match

# foo bar baz
# 2NGRzWMVOol86
# UjAhcC7QgIpoj
