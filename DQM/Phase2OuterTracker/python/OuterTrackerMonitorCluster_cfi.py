import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorCluster = cms.EDAnalyzer('OuterTrackerMonitorCluster',
    
    TopFolderName = cms.string('Phase2OuterTracker'),

# Number of clusters per layer
    TH1TTCluster_Stack = cms.PSet(
        Nbinsx = cms.int32(6),
        xmax = cms.double(6.5),                      
        xmin = cms.double(0.5)
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
          
)
