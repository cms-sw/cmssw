import FWCore.ParameterSet.Config as cms

EgammaZJetToMuPlusProbe = cms.EDFilter("EgammaProbeSelector",
    JetEtaMax = cms.double(2.7),
    JetEtaMin = cms.double(-2.7),
    ScEtaMin = cms.double(-2.7),
    SuperClusterEndCapCollection = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    MinNumberOfSuperClusters = cms.int32(1),
    MinNumberOfJets = cms.int32(1),
    ScEtMin = cms.double(5.0),
    SuperClusterBarrelCollection = cms.InputTag("correctedHybridSuperClusters"),
    JetEtMin = cms.double(9999999.0),
    JetCollection = cms.InputTag("ak4CaloJets"),
    ScEtaMax = cms.double(2.7)
)


