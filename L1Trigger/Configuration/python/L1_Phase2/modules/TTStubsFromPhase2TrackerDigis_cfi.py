import FWCore.ParameterSet.Config as cms

TTStubsFromPhase2TrackerDigis = cms.EDProducer("TTStubBuilder_Phase2TrackerDigi_",
    BarrelMax10GLay = cms.uint32(2),
    CBClimit = cms.uint32(3),
    FEineffs = cms.bool(False),
    MPAlimit = cms.uint32(5),
    OnlyOnePerInputCluster = cms.bool(True),
    PS10GCIClimit = cms.uint32(35),
    PS5GCIClimit = cms.uint32(16),
    SS5GCIClimit = cms.uint32(16),
    TEDD1Max10GRing = cms.uint32(7),
    TEDD2Max10GRing = cms.uint32(3),
    TTClusters = cms.InputTag("TTClustersFromPhase2TrackerDigis","ClusterInclusive")
)
