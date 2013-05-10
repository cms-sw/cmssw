import FWCore.ParameterSet.Config as cms

ecalEndcapRecoSummary = cms.EDAnalyzer("EERecoSummary",
    prefixME = cms.untracked.string('EcalEndcap'),    
    superClusterCollection_EE = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    basicClusterCollection_EE = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters"),
    recHitCollection_EE       = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    redRecHitCollection_EE    = cms.InputTag("reducedEcalRecHitsEE"),
                                    
    ethrEE = cms.double(1.2),

    scEtThrEE = cms.double(0.0),
)

