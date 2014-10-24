import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorStub = cms.EDAnalyzer('OuterTrackerMonitorStub',
    
    TopFolderName = cms.string('OuterTracker'),

# Number of Stubs per layer
    TH1TTStub_Stack = cms.PSet(
        Nbinsx = cms.int32(6),
        xmax = cms.double(6.5),                      
        xmin = cms.double(0.5)
        ),
    
# Stub eta distribution
    TH1TTStub_Eta = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(3.0),                      
        xmin = cms.double(-3.0)
        ),
    
# Stub Width vs. I/O sensor
    TH2TTStub_Width = cms.PSet(
        Nbinsx = cms.int32(10),
        xmax = cms.double(9.5),                      
        xmin = cms.double(-0.5),
        Nbinsy = cms.int32(2),
        ymax = cms.double(1.5),                      
        ymin = cms.double(-0.5)
        ),
          
)
