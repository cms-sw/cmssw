import FWCore.ParameterSet.Config as cms

#
#  module for filtering of rechits. user provides noise threshold in GeV units
#  Author: Shahram Rahatlou, University of Rome & INFN
#  $Id: egammaRechitFilter.cfi,v 1.1 2006/05/25 10:42:33 rahatlou Exp $
#
rechitFilter = cms.EDFilter("RecHitFilter",
    noiseThreshold = cms.double(0.06),
    hitProducer = cms.string('ecalrechit'),
    hitCollection = cms.string('EcalRecHitsEB'),
    reducedHitCollection = cms.string('FilteredEcalRecHitCollection')
)


