import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ECFAdder_cfi import ECFAdder

ecf = ECFAdder.clone(
             src = cms.InputTag("ak8PFJetsPuppi"),
             ecftype = cms.string("ECF")
             )

ecfCbeta1 = ecf.clone(
             ecftype = cms.string("C")
             )

ecfCbeta2 = ecfCbeta1.clone(
             alpha = cms.double(2.0),
             beta = cms.double(2.0)
             )

ecfDbeta1 = ecf.clone(
             ecftype = cms.string("D"),
             Njets = cms.vuint32(2)
             )

ecfDbeta2 = ecfDbeta1.clone(
             alpha = cms.double(2.0),
             beta = cms.double(2.0)
             )


ecfMbeta1 = ecf.clone(
             ecftype = cms.string("M")
             )

ecfMbeta2 = ecfMbeta1.clone(
             alpha = cms.double(2.0),
             beta = cms.double(2.0)
             )

ecfNbeta1 = ecf.clone(
             ecftype = cms.string("N")
             )

ecfNbeta2 = ecfNbeta1.clone(
             alpha = cms.double(2.0),
             beta = cms.double(2.0)
             )

ecfUbeta1 = ecf.clone(
             ecftype = cms.string("U")
             )

ecfUbeta2 = ecfUbeta1.clone(
             alpha = cms.double(2.0),
             beta = cms.double(2.0)
             )

