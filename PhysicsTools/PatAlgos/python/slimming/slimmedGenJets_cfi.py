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

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
(pp_on_AA_2018 | pp_on_PbPb_run3).toModify(slimmedGenJets, src = "ak4HiSignalGenJets")
(pp_on_AA_2018 | pp_on_PbPb_run3).toModify(slimmedGenJetsAK8, cut = 'pt>9999', nLoose = 0)
from Configuration.ProcessModifiers.genJetSubEvent_cff import genJetSubEvent
genJetSubEvent.toModify(slimmedGenJets, src = "ak4HiGenJetsCleaned")
