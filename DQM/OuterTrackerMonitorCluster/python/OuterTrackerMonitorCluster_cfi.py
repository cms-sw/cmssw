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

# Cluster Stacks
    TH1TTCluster_Stack = cms.PSet(
        Nbinsx = cms.int32(12),
        xmax = cms.double(11.5),                      
        xmin = cms.double(-0.5)
        ),

# Cluster Width vs. I/O sensor
    TH2TTCluster_Width = cms.PSet(
        Nbinsx = cms.int32(10),
        xmax = cms.double(9.5),                      
        xmin = cms.double(-0.5),
        Nbinsy = cms.int32(2),
        ymax = cms.double(1.5),                      
        ymin = cms.double(-0.5)
        ),
          
)
