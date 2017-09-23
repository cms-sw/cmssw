import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ECFAdder_cfi import ECFAdder

ecf = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("ECF")
             )

ecfCbeta1 = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("C")
             )

ecfCbeta2 = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("C"),
             alpha = cms.double(2.0),
             beta = cms.double(2.0)
             )

ecfDbeta1 = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("D"),
             Njets = cms.vuint32(2)
             )

ecfDbeta2 = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("D"),
             Njets = cms.vuint32(2),
             alpha = cms.double(2.0),
             beta = cms.double(2.0)
             )


ecfMbeta1 = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("M")
             )

ecfMbeta2 = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("M"),
             alpha = cms.double(2.0),
             beta = cms.double(2.0)
             )

ecfNbeta1 = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("N")
             )

ecfNbeta2 = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("N"),
             alpha = cms.double(2.0),
             beta = cms.double(2.0)
             )

ecfUbeta1 = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("U")
             )

ecfUbeta2 = ECFAdder.clone(
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("U"),
             alpha = cms.double(2.0),
             beta = cms.double(2.0)
             )

