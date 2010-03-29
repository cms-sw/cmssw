import FWCore.ParameterSet.Config as cms

#
#  module for filtering of rechits. user provides noise threshold in GeV units
#  Author: Shahram Rahatlou, University of Rome & INFN
#  $Id: egammaRechitFilter_cfi.py,v 1.2 2008/04/21 03:24:01 rpw Exp $
#
rechitFilter = cms.EDProducer("RecHitFilter",
    noiseThreshold = cms.double(0.06),
    hitProducer = cms.string('ecalrechit'),
    hitCollection = cms.string('EcalRecHitsEB'),
    reducedHitCollection = cms.string('FilteredEcalRecHitCollection')
)


