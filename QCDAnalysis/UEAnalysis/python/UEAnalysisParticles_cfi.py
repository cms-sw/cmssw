import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
# select charged GenParticles with pt > 0.29
# 
# (threshold used to be 0.89, but tracking in 2_0_0 
#  saves tracks with pT > 0.29)
goodParticles = cms.EDFilter("GenParticleSelector",
    filter = cms.bool(False),
    src = cms.InputTag("genParticles"),
    cut = cms.string('pt > 0.0'),
    stableOnly = cms.bool(True)
)

chargeParticles = cms.EDFilter("GenParticleSelector",
    filter = cms.bool(False),
    src = cms.InputTag("genParticles"),
    cut = cms.string('charge != 0 & pt > 0.29'),
    stableOnly = cms.bool(True)
)

UEAnalysisParticles = cms.Sequence(genJetParticles*goodParticles*chargeParticles)

