import FWCore.ParameterSet.Config as cms

# HGCAL producer of rechits starting from digis
HGCalUncalibRecHit = cms.EDProducer("HGCalUncalibRecHitProducer",
    HGCEEdigiCollection = cms.InputTag("HGCalDigis","HGCEEDigis"),
    HGCEEhitCollection = cms.string('HGCEEUncalibRecHits'),
    HGCHEFdigiCollection = cms.InputTag("HGCalDigis","HGCHEFDigis"),
    HGCHEFhitCollection = cms.string('HGCHEFUncalibRecHits'),
    HGCHEBdigiCollection = cms.InputTag("HGCalDigis","HGCHEBDigis"),
    HGCHEBhitCollection = cms.string('HGCHEBUncalibRecHits'),                                    
    algo = cms.string("HGCalUncalibRecHitWorkerWeights")
)
