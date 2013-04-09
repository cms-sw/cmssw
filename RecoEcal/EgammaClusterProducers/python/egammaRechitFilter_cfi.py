import FWCore.ParameterSet.Config as cms

#
#  module for filtering of rechits. user provides noise threshold in GeV units
#  Author: Shahram Rahatlou, University of Rome & INFN
#  $Id: egammaRechitFilter_cfi.py,v 1.3 2010/03/01 21:25:44 wmtan Exp $
#
rechitFilter = cms.EDProducer("RecHitFilter",
    noiseEnergyThreshold = cms.double(0.08),
    noiseChi2Threshold = cms.double(40),
    hitProducer = cms.string(''),
    hitCollection = cms.string('EcalRecHitsEB'),
    reducedHitCollection = cms.string('FilteredEcalRecHitCollection')
)


