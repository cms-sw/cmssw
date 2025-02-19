import FWCore.ParameterSet.Config as cms

SiStripRoI = cms.EDProducer("SiStripRawToClustersRoI",
    # layers of interest
    Layers = cms.untracked.int32(-1),
    BJetEtaWindow = cms.untracked.double(0.2),
    TauJetEtaWindow = cms.untracked.double(0.5),
    Random = cms.untracked.bool(False),
    MuonL2 = cms.InputTag("hltL2Muons"),
    ElectronEndcapL2 = cms.InputTag("hltIslandSuperClustersL1Isolated","islandEndcapSuperClusters"),
    ElectronPhiWindow = cms.untracked.double(0.16),
    TauJetPhiWindow = cms.untracked.double(0.5),
    # define objects of interest
    Global = cms.untracked.bool(True),
    ElectronBarrelL2 = cms.InputTag("hltHybridSuperClustersL1Isolated"),
    BJetL2 = cms.InputTag("iterativeCone5CaloJets"),
    TauJets = cms.untracked.bool(False),
    # define tracker windows
    ElectronEtaWindow = cms.untracked.double(0.16),
    Electrons = cms.untracked.bool(False),
    TauJetL2 = cms.InputTag("l2TauJetsProvider","SingleTau"),
    MuonPhiWindow = cms.untracked.double(0.16),
    # define input tags
    SiStripLazyGetter = cms.InputTag("SiStripRawToClustersFacility"),
    Muons = cms.untracked.bool(False),
    BJets = cms.untracked.bool(False),
    MuonEtaWindow = cms.untracked.double(0.16),
    BJetPhiWindow = cms.untracked.double(0.2)
)


