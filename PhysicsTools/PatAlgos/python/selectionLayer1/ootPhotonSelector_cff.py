import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.selectionLayer1.photonSelector_cfi import *

selectedPatOOTPhotons = selectedPatPhotons.clone()
selectedPatOOTPhotons.src = cms.InputTag("patOOTPhotons")
# foo bar baz
# 9j90UPefjWIG6
# M00PTod3JTPyT
