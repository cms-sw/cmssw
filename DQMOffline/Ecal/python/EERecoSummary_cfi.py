import FWCore.ParameterSet.Config as cms

ecalEndcapRecoSummary = cms.EDAnalyzer("EERecoSummary",
    prefixME = cms.untracked.string('EcalEndcap'),    
    superClusterCollection_EE = cms.InputTag("particleFlowSuperClusterECAL", "particleFlowSuperClusterECALEndcapWithPreshower"),
    basicClusterCollection_EE = cms.InputTag("particleFlowClusterECAL"),
    recHitCollection_EE       = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    redRecHitCollection_EE    = cms.InputTag("reducedEcalRecHitsEE"),
                                    
    ethrEE = cms.double(1.2),

    scEtThrEE = cms.double(0.0),
)

