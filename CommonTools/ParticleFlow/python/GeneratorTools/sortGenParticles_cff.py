import FWCore.ParameterSet.Config as cms
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

decaysFromZs = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop *  ", # this is the default
    "keep+ pdgId = {Z0}",
    "drop pdgId = {Z0}"
    )
)

sortGenParticlesSequence = cms.Sequence(
    decaysFromZs
    )
