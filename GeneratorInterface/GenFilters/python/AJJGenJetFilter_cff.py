import FWCore.ParameterSet.Config as cms

from GeneratorInterface.GenFilters.AJJGenJetFilter_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

vjjGenJetFilterSeq = cms.Sequence(
  genParticlesForJetsNoNu*
  ak4GenJetsNoNu*
  vjjGenJetFilterPhotonInBarrelMjj300
)

