import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import *

##################################################################################
#
# Heavy Ion pat::Muon Production
#

muonMatch.matched = cms.InputTag("hiGenParticles")
allLayer1Muons.embedGenMatch = cms.bool(True)

hiPatMuonSequence = cms.Sequence( muonMatch * allLayer1Muons )

##################################################################################
#
# Heavy Ion pat::Muon Selection
#


selectedLayer1Muons.cut = cms.string('pt > 0. & abs(eta) < 12.')
