import FWCore.ParameterSet.Config as cms

ecalBarrelRecoSummary = cms.EDAnalyzer("EBRecoSummary",
    prefixME = cms.untracked.string('EcalBarrel'),    
    superClusterCollection_EB = cms.InputTag("particleFlowSuperClusterECAL", "particleFlowSuperClusterECALBarrel"),
    recHitCollection_EB       = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    redRecHitCollection_EB    = cms.InputTag("reducedEcalRecHitsEB"),
    basicClusterCollection_EB = cms.InputTag("particleFlowClusterECAL"),
                                    
    ethrEB = cms.double(0.8),

    scEtThrEB = cms.double(0.0),
)

