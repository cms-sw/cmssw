import FWCore.ParameterSet.Config as cms

#
#  Author: N. Marinelli, U. of Notre Dame, US
#  $Id: simplePhotonAnalyzer_cfi.py,v 1.4 2009/11/26 19:37:36 nancy Exp $ 
#
simplePhotonAnalyzer = cms.EDAnalyzer("SimplePhotonAnalyzer",
    phoProducer = cms.string('photons'),
    mcProducer = cms.string('generator'),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    # 1=pt10,2=pt35,3=Hgg, 4=gamJetHighEnergy
    sample = cms.int32(2),
    photonCollection = cms.string(''),
    primaryVertexProducer = cms.string('offlinePrimaryVerticesWithBS'),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


