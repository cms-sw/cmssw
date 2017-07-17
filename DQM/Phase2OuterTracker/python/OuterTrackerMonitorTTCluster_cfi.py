import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorTTCluster = cms.EDAnalyzer('OuterTrackerMonitorTTCluster',

    TopFolderName = cms.string('Phase2OuterTracker'),
    TTClusters    = cms.InputTag("TTClustersFromPhase2TrackerDigis", "ClusterInclusive"),
#    TTClusters    = cms.InputTag("TTStubsFromPhase2TrackerDigis", "ClusterAccepted"),

# Number of clusters per layer
    TH1TTCluster_Barrel = cms.PSet(
        Nbinsx = cms.int32(6),
        xmax = cms.double(6.5),                  
        xmin = cms.double(0.5)
        ),

# Number of clusters per disc
    TH1TTCluster_ECDiscs = cms.PSet(
        Nbinsx = cms.int32(5),
        xmax = cms.double(5.5),                 
        xmin = cms.double(0.5)
        ),

# Number of clusters per EC ring
    TH1TTCluster_ECRings = cms.PSet(
        Nbinsx = cms.int32(16),
        xmin = cms.double(0.5),
        xmax = cms.double(16.5)
        ),

# Cluster eta distribution
    TH1TTCluster_Eta = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3.0),
        xmin = cms.double(-3.0)
        ),

# Cluster Width vs. I/O sensor
    TH2TTCluster_Width = cms.PSet(
        Nbinsx = cms.int32(7),
        xmax = cms.double(6.5),                   
        xmin = cms.double(-0.5),
        Nbinsy = cms.int32(2),
        ymax = cms.double(1.5),
        ymin = cms.double(-0.5)
        ),

# TTCluster barrel y vs x
# TTCluster forward/backward endcap y vs x
    TH2TTCluster_Position = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(120),
        xmin = cms.double(-120),
        Nbinsy = cms.int32(960),
        ymax = cms.double(120),
        ymin = cms.double(-120)
        ),

#TTCluster barrel y vs x zoomed
    TH2TTCluster_Barrel_XY_Zoom = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(60),
        xmin = cms.double(30),
        Nbinsy = cms.int32(960),
        ymax = cms.double(15),
        ymin = cms.double(-15)
        ),

#TTCluster #rho vs z
    TH2TTCluster_RZ = cms.PSet(
        Nbinsx = cms.int32(900),
        xmax = cms.double(300),
        xmin = cms.double(-300),
        Nbinsy = cms.int32(900),
        ymax = cms.double(120),
        ymin = cms.double(0)
        ),

#TTCluster Forward Endcap #rho vs. z
    TH2TTCluster_Endcap_Fw_RZ_Zoom = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(170),
        xmin = cms.double(140),
        Nbinsy = cms.int32(960),
        ymax = cms.double(60),
        ymin = cms.double(30)
        ),

#TTCluster Backward Endcap #rho vs. z
    TH2TTCluster_Endcap_Bw_RZ_Zoom = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(-140),
        xmin = cms.double(-170),
        Nbinsy = cms.int32(960),
        ymax = cms.double(100),
        ymin = cms.double(70)
        )

)
