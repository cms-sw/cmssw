import FWCore.ParameterSet.Config as cms

# HGCAL producer of rechits starting from digis
HGCalUncalibRecHit = cms.EDProducer("HGCalUncalibRecHitProducer",
    HGCEEdigiCollection = cms.InputTag("mix","HGCDigisEE"),
    HGCEEhitCollection = cms.string('HGCEEUncalibRecHits'),
    HGCHEFdigiCollection = cms.InputTag("mix","HGCDigisHEfront"),
    HGCHEFhitCollection = cms.string('HGCHEFUncalibRecHits'),
    HGCHEBdigiCollection = cms.InputTag("mix","HGCDigisHEback"),
    HGCHEBhitCollection = cms.string('HGCHEBUncalibRecHits'),                                    
    algo = cms.string("HGCalUncalibRecHitWorkerWeights")
)
