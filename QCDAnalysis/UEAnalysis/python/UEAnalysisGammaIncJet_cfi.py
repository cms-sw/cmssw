import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
goodParticles = cms.EDFilter("GenParticleSelector",
    filter = cms.bool(False),
    src = cms.InputTag("genParticles"),
    cut = cms.string('pt > 0.0 & status = 1')
)

chargeParticles = cms.EDFilter("GenParticleSelector",
    filter = cms.bool(False),
    src = cms.InputTag("genParticles"),
    cut = cms.string('pt > 0.29 & status = 1')
)

gammaParticles = cms.EDFilter("GenParticleSelector",
    filter = cms.bool(False),
    src = cms.InputTag("genParticles"),
    cut = cms.string('pdgId = 22')
)

UEAnalysisGammaIncJet = cms.Sequence(genParticles*genJetParticles*goodParticles*gammaParticles*chargeParticles)
