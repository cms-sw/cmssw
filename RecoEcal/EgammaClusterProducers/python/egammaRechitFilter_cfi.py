import FWCore.ParameterSet.Config as cms

#
#  module for filtering of rechits. user provides noise threshold in GeV units
#  Author: Shahram Rahatlou, University of Rome & INFN
#  $Id: egammaRechitFilter_cfi.py,v 1.5 2013/04/19 07:24:05 argiro Exp $
#
rechitFilter = cms.EDProducer("RecHitFilter",
    noiseEnergyThreshold = cms.double(0.08),
    noiseChi2Threshold = cms.double(40),
    hitCollection = cms.InputTag('EcalRecHit','EcalRecHitsEB'),
    reducedHitCollection = cms.string('FilteredEcalRecHitCollection')
)


