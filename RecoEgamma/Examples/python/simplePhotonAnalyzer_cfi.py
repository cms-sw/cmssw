import FWCore.ParameterSet.Config as cms

#
#  Author: N. Marinelli, U. of Notre Dame, US
#  $Id: simplePhotonAnalyzer.cfi,v 1.8 2008/06/03 13:53:55 nancy Exp $ 
#
simplePhotonAnalyzer = cms.EDAnalyzer("SimplePhotonAnalyzer",
    phoProducer = cms.string('photons'),
    mcProducer = cms.string('source'),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    # 1=pt10,2=pt35,3=E10,4=E1000
    sample = cms.int32(2),
    photonCollection = cms.string(''),
    primaryVertexProducer = cms.string('offlinePrimaryVerticesWithBS'),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


