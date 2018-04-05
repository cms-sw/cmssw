import FWCore.ParameterSet.Config as cms

TTStubsFromPhase2TrackerDigis = cms.EDProducer("TTStubBuilder_Phase2TrackerDigi_",
    TTClusters = cms.InputTag("TTClustersFromPhase2TrackerDigis", "ClusterInclusive"),
    OnlyOnePerInputCluster = cms.bool(True), 
    FEineffs      = cms.bool(False), # Turn ON (true) or OFF (false) the dynamic FE inefficiency accounting
    CBClimit      = cms.uint32(3),   # CBC chip limit (in stubs/chip/BX)
    MPAlimit      = cms.uint32(5),   # MPA chip limit (in stubs/chip/2BX)
    SS5GCIClimit  = cms.uint32(16),  # 2S 5G chip limit (in stubs/CIC/8BX)
    PS5GCIClimit  = cms.uint32(17),  # PS 5G chip limit (in stubs/CIC/8BX)
    PS10GCIClimit = cms.uint32(35),  # PS 10G chip limit (in stubs/CIC/8BX)
    TEDD1Max10GRing = cms.uint32(3), # How many rings are PS10G in TEDD1 disks (1/2)? 
    TEDD2Max10GRing = cms.uint32(0)  # How many rings are PS10G in TEDD2 disks (3/4/5)? 
)


