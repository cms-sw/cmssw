import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorStub = cms.EDAnalyzer('OuterTrackerMonitorStub',
    
    TopFolderName = cms.string('OuterTracker'),

# TTStub barrel y vs x
    TH2TTStub_Barrel_XY = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(120),                      
        xmin = cms.double(-120),
        Nbinsy = cms.int32(960),
        ymax = cms.double(120),                      
        ymin = cms.double(-120)
        ),
          
)
