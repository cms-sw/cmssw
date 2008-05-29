import FWCore.ParameterSet.Config as cms

caloRecoTauTagInfoProducer = cms.EDProducer("CaloRecoTauTagInfoProducer",
    tkminTrackerHitsn = cms.int32(8),
    tkminPixelHitsn = cms.int32(2),
    ECALBasicClusterpropagTrack_matchingDRConeSize = cms.double(0.015),
    #string PVProducer                         = "pixelVertices"
    PVProducer = cms.string('offlinePrimaryVertices'),
    # parameters of the considered rec. Tracks (were catched through a JetTracksAssociation object) :
    tkminPt = cms.double(1.0),
    ESRecHitsSource = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    UsePVconstraint = cms.bool(False),
    tkmaxChi2 = cms.double(100.0),
    # parameters of the considered EcalRecHits
    EBRecHitsSource = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    BarrelBasicClustersSource = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    EndcapBasicClustersSource = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),

    smearedPVsigmaY = cms.double(0.0015),
    smearedPVsigmaX = cms.double(0.0015),
    ECALBasicClusterminE = cms.double(1.0),
    smearedPVsigmaZ = cms.double(0.005),
    tkPVmaxDZ = cms.double(0.2), ##considered if UsePVconstraint=true 

    # parameters of the considered neutral ECAL BasicClusters
    ECALBasicClustersAroundCaloJet_DRConeSize = cms.double(0.5),
    tkmaxipt = cms.double(0.03),
    EERecHitsSource = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    CaloJetTracksAssociatorProducer = cms.string('ic5JetTracksAssociatorAtVertex')
)


