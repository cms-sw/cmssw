import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.GeneratorTools.genMetTrue_cff  import *

#not necessary?
from PhysicsTools.PFCandProducer.GeneratorTools.sortGenParticles_cff import *

# To reconstruct genjets without the neutrinos
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *
genParticlesForJets.ignoreParticleIDs.append(14)
genParticlesForJets.ignoreParticleIDs.append(12)
genParticlesForJets.ignoreParticleIDs.append(16)
genParticlesForJets.excludeResonances = False

ak5GenJetsNoNu.src = "genParticlesForJets"
iterativeCone5GenJetsNoNu.src = "genParticlesForJets"
ak7GenJetsNoNu.src = "genParticlesForJets"


genForPF2PATSequence = cms.Sequence(
    #MB genMetTrueSequence + 
    genJetParticles +  
    ak5GenJetsNoNu +
    ak7GenJetsNoNu +
    iterativeCone5GenJetsNoNu
)
