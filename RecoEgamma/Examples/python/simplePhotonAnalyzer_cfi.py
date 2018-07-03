import FWCore.ParameterSet.Config as cms

#
#  Author: N. Marinelli, U. of Notre Dame, US
#
simplePhotonAnalyzer = cms.EDAnalyzer("SimplePhotonAnalyzer",
    phoProducer = cms.string('photons'),
    mcProducer = cms.string('generatorSmeared'),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    pfEgammaCandidates = cms.InputTag("particleFlowEGamma"),                                  
    # 1=pt10,2=pt35,3=Hgg, 4=gamJetHighEnergy
    sample = cms.int32(2),
    photonCollection = cms.string(''),
    primaryVertexProducer = cms.string('offlinePrimaryVerticesWithBS'),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    valueMapPhotons = cms.string("valMapAssociationPFEgammaCandidateToPhoton"),                                               
)


