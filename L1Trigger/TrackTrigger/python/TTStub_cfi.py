import FWCore.ParameterSet.Config as cms

TTStubsFromPhase2TrackerDigis = cms.EDProducer("TTStubBuilder_Phase2TrackerDigi_",
    TTClusters = cms.InputTag("TTClustersFromPhase2TrackerDigis", "ClusterInclusive"),
    OnlyOnePerInputCluster = cms.bool(True), 
    # Warning: results if FEineffs=True depend on order events processed in,
    # so in multithreaded job can change if same job run twice.
    FEineffs      = cms.bool(False), # Turn ON (true) or OFF (false) dynamic FE stub truncation
    CBClimit      = cms.uint32(3),   # CBC chip limit (in stubs/chip/BX)
    MPAlimit      = cms.uint32(5),   # MPA chip limit (in stubs/chip/2BX)
    # N.B. CIC chip uses FEC5 mode for PS modules & FEC12 mode for 2S modules.
    SS5GCIClimit  = cms.uint32(16),  # 2S 5G chip limit (in stubs/CIC/8BX)
    PS5GCIClimit  = cms.uint32(16),  # PS 5G chip limit (in stubs/CIC/8BX)
    PS10GCIClimit = cms.uint32(35),  # PS 10G chip limit (in stubs/CIC/8BX)
    # N.B. CMSSW defines inner ring present in any given wheel as number 1.
    TEDD1Max10GRing = cms.uint32(7), # No. of PS10G rings in TEDD1 disks (1/2)
    TEDD2Max10GRing = cms.uint32(3), # No. of PS10G rings in TEDD2 disks (3/4/5) 
    BarrelMax10GLay = cms.uint32(2)  # No. of PS10G layers in barrel                                               
)


