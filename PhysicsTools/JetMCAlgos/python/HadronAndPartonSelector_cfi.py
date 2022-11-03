import FWCore.ParameterSet.Config as cms

# select hadrons and partons for the jet flavour
selectedHadronsAndPartons = cms.EDProducer('HadronAndPartonSelector',
    src = cms.InputTag("generator"),
    particles = cms.InputTag("genParticles"),
    partonMode = cms.string("Auto"),
    CheckHerwig7Flag = cms.bool(False),
    fullChainPhysPartons = cms.bool(True)
)
# select hadrons and partons for the slimmedGenJetsFlavourInfos, required for origin identification

selectedHadronsAndPartonsForGenJetsFlavourInfos = selectedHadronsAndPartons.clone(particles = "prunedGenParticles")

from Configuration.Eras.Modifier_run2_JMENanoHerwig7_cff import Modifier_run2_JMENanoHerwig7
Modifier_run2_JMENanoHerwig7.toModify( selectedHadronsAndPartons, CheckHerwig7Flag = True )
