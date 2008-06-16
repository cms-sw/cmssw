import FWCore.ParameterSet.Config as cms

#
# Configure HLT filter and alcareco stream for  
# Ecal PhiSymmetry
#
# Author: Stefano Argiro
# $Id: HLTEcalPhiSymFilter.cfi,v 1.4 2008/06/12 12:46:04 argiro Exp $
alCaPhiSymStream = cms.EDFilter("HLTEcalPhiSymFilter",
    endcapHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    eCut_barrel = cms.double(0.15),
    eCut_endcap = cms.double(0.65),
    phiSymBarrelHitCollection = cms.string('phiSymEcalRecHitsEB'),
    barrelHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    phiSymEndcapHitCollection = cms.string('phiSymEcalRecHitsEE')
)


