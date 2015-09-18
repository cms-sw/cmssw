import FWCore.ParameterSet.Config as cms
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

decaysFromZs = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop *  ", # this is the default
    "keep+ obj.pdgId() == {Z0}",
    "drop  obj.pdgId() == {Z0}"
    )
)

sortGenParticlesSequence = cms.Sequence(
    decaysFromZs
    )
