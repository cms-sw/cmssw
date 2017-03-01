import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff import *
from PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff import *
from PhysicsTools.PatAlgos.slimming.slimming_cff import *
from RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi import *

miniAOD=cms.Sequence()
