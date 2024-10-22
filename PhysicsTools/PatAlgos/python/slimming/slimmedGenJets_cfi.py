import FWCore.ParameterSet.Config as cms

slimmedGenJets = cms.EDProducer("PATGenJetSlimmer",
    src = cms.InputTag("ak4GenJetsNoNu"),
    packedGenParticles = cms.InputTag("packedGenParticles"),
    cut = cms.string("pt > 8"),
    cutLoose = cms.string(""),
    nLoose = cms.uint32(0),
    clearDaughters = cms.bool(False), #False means rekeying
    dropSpecific = cms.bool(False),
)

slimmedGenJetsAK8 = cms.EDProducer("PATGenJetSlimmer",
    src = cms.InputTag("ak8GenJetsNoNu"),
    packedGenParticles = cms.InputTag("packedGenParticles"),
    cut = cms.string("pt > 80"),
    cutLoose = cms.string("pt > 10."),
    nLoose = cms.uint32(3),
    clearDaughters = cms.bool(False), #False means rekeying
    dropSpecific = cms.bool(False),
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(slimmedGenJets, src = "ak4HiSignalGenJets")
pp_on_AA.toModify(slimmedGenJetsAK8, cut = 'pt>9999', nLoose = 0)
from Configuration.ProcessModifiers.genJetSubEvent_cff import genJetSubEvent
genJetSubEvent.toModify(slimmedGenJets, src = "ak4HiGenJetsCleaned")
