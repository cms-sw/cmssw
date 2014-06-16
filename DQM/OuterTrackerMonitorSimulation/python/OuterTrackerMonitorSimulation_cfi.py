import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorSimulation = cms.EDAnalyzer('OuterTrackerMonitorSimulation',
                                     
    ClusterProducerStrip = cms.InputTag('siStripClusters'),
    
    TopFolderName = cms.string('OuterTracker'),
    

# TPart Pt
    TH1TPart_Pt = cms.PSet(
        Nbinsx = cms.int32(25),
        xmax = cms.double(50.0),                      
        xmin = cms.double(0.0)
        ),

# TPart Eta/Phi
    TH1TPart_Angle_Pt10 = cms.PSet(
        Nbinsx = cms.int32(60),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416)
        ),

# Sim Vertex XY
    TH1SimVtx_XY = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(0.4),                      
        xmin = cms.double(-0.4),
				Nbinsy = cms.int32(100),
				ymax = cms.double(0.4),                      
        ymin = cms.double(-0.4)
        ),

# Sim Vertex RZ
    TH1SimVtx_RZ = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(50.0),                      
        xmin = cms.double(-50.0),
				Nbinsy = cms.int32(100),
				ymax = cms.double(0.4),                      
        ymin = cms.double(0.0)
        ),

# CW vs. TPart AbsEta
    TH1TPart_AbsEta_CW = cms.PSet(
        Nbinsx = cms.int32(30),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(0.0)
        ),

# CW vs. TPart Eta
    TH1TPart_Eta_CW = cms.PSet(
        Nbinsx = cms.int32(60),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416)
        ),

# Cluster Stacks
    TH1TTCluster_Stack = cms.PSet(
        Nbinsx = cms.int32(12),
        xmax = cms.double(11.5),                      
        xmin = cms.double(-0.5)
        ),
          
)
