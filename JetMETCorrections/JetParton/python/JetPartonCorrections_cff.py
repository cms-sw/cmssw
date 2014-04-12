import FWCore.ParameterSet.Config as cms

# File: JetCorrections.cff
# Author: O. Kodolova
# Date: 1/24/07
#
# JetParton corrections for the icone5, icone7 jets. (ORCA derived)
# 
JetPartonCorrectorIcone5 = cms.ESSource("JetPartonCorrectionService",
    MixtureType = cms.int32(3),
    Radius = cms.double(0.5),
    tagName = cms.string('PartonScale_IterativeCone0.5.txt'),
    label = cms.string('JetPartonCorrectorIcone5')
)

es_prefer_JetPartonCorrectorIcone5 = cms.ESPrefer("JetPartonCorrectionService","JetPartonCorrectorIcone5")
JetPartonCorrectorIcone7 = cms.ESSource("JetPartonCorrectionService",
    MixtureType = cms.int32(3),
    Radius = cms.double(0.7),
    tagName = cms.string('PartonScale_IterativeCone0.7.txt'),
    label = cms.string('JetPartonCorrectorIcone7')
)

JetPartonCorJetIcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('JetPartonCorrectorIcone5'),
    alias = cms.untracked.string('JetPartonCorJetIcone5')
)

JetPartonCorJetIcone7 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("iterativeCone7CaloJets"),
    correctors = cms.vstring('JetPartonCorrectorIcone7'),
    alias = cms.untracked.string('JetPartonCorJetIcone7')
)

JetPartonCorrections = cms.Sequence(JetPartonCorJetIcone5*JetPartonCorJetIcone7)

