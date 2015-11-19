import FWCore.ParameterSet.Config as cms

from GeneratorInterface.GenFilters.VBFGenJetFilter_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

vbfGenJetFilterASeq = cms.Sequence(
  genParticlesForJetsNoNu*
  ak4GenJetsNoNu*
  vbfGenJetFilterA
)

vbfGenJetFilterBSeq = cms.Sequence(
  genParticlesForJetsNoNu*
  ak4GenJetsNoNu*
  vbfGenJetFilterB
)

vbfGenJetFilterCSeq = cms.Sequence(
  genParticlesForJetsNoNu*
  ak4GenJetsNoNu*
  vbfGenJetFilterC
)
