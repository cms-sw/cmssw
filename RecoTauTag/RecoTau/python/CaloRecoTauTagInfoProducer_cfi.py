import FWCore.ParameterSet.Config as cms

caloRecoTauTagInfoProducer = cms.EDProducer("CaloRecoTauTagInfoProducer",
    ECALBasicClusterpropagTrack_matchingDRConeSize = cms.double(0.015),
    #string PVProducer = "pixelVertices"
    PVProducer = cms.InputTag('offlinePrimaryVertices'),
    # parameters of the considered rec. Tracks (were catched through a JetTracksAssociation object) :
    tkminPt = cms.double(0.5),
    tkminPixelHitsn = cms.int32(0),
    tkminTrackerHitsn = cms.int32(3),	
    tkmaxChi2 = cms.double(100.0),	
    UsePVconstraint = cms.bool(True),

    UseTrackQuality = cms.bool(True),
    #only used if UseTrackQuality is True
    tkQuality = cms.string('highPurity'),
    
    BarrelBasicClustersSource = cms.InputTag("hybridSuperClusters", "hybridBarrelBasicClusters"),
    EndcapBasicClustersSource = cms.InputTag("multi5x5SuperClusters", "multi5x5EndcapBasicClusters"),

    smearedPVsigmaY = cms.double(0.0015),
    smearedPVsigmaX = cms.double(0.0015),
    ECALBasicClusterminE = cms.double(1.0),
    smearedPVsigmaZ = cms.double(0.005),
    tkPVmaxDZ = cms.double(1.0), ##considered if UsePVconstraint=true 

    # parameters of the considered neutral ECAL BasicClusters
    ECALBasicClustersAroundCaloJet_DRConeSize = cms.double(0.5),
    tkmaxipt = cms.double(0.1),
    CaloJetTracksAssociatorProducer = cms.InputTag('ic5JetTracksAssociatorAtVertex')
)


