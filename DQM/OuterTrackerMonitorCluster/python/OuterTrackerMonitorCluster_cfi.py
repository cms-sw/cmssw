import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorCluster = cms.EDAnalyzer('OuterTrackerMonitorCluster',
                                     
    ClusterProducerStrip = cms.InputTag('siStripClusters'),
    
    TopFolderName = cms.string('OuterTracker'),
               
# Number of Cluster in Strip
    TH1NClusStrip = cms.PSet(
        Nbinsx = cms.int32(500),
        xmax = cms.double(99999.5),                      
        xmin = cms.double(-0.5)
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
