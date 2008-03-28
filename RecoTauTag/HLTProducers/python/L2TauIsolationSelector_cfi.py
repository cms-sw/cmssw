import FWCore.ParameterSet.Config as cms

L2TauIsolationSelector = cms.EDFilter("L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag("doubleTauL2Producer","L2TauIsolationInfoAssociator"),
    SeedTowerEt = cms.double(0.0),
    ClusterEtaRMS = cms.double(1000.0),
    #Put Large Values for Tower/Cluster cuts
    TowerIsolEt = cms.double(1000.0),
    ECALIsolEt = cms.double(3.0),
    ClusterDRRMS = cms.double(1000.0),
    ClusterPhiRMS = cms.double(1000.0),
    ClusterNClusters = cms.int32(1000),
    MinJetEt = cms.double(15.0)
)


