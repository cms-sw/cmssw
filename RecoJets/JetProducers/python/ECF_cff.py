import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ECFAdder_cfi import ECFAdder

ecf = ECFAdder.clone(
             src     = "ak8PFJetsPuppi",
             ecftype = "ECF"
             )

ecfCbeta1 = ecf.clone(
             ecftype = "C"
             )

ecfCbeta2 = ecfCbeta1.clone(
             alpha = 2.0,
             beta  = 2.0
             )

ecfDbeta1 = ecf.clone(
             ecftype = "D",
             Njets   = [2]
             )

ecfDbeta2 = ecfDbeta1.clone(
             alpha = 2.0,
             beta  = 2.0
             )

ecfMbeta1 = ecf.clone(
             ecftype = "M"
             )

ecfMbeta2 = ecfMbeta1.clone(
             alpha = 2.0,
             beta  = 2.0
             )

ecfNbeta1 = ecf.clone(
             ecftype = "N"
             )

ecfNbeta2 = ecfNbeta1.clone(
             alpha = 2.0,
             beta  = 2.0
             )

ecfUbeta1 = ecf.clone(
             ecftype = "U"
             )

ecfUbeta2 = ecfUbeta1.clone(
             alpha = 2.0,
             beta  = 2.0
             )
