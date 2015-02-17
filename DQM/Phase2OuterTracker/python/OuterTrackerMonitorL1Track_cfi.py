import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorL1Track = cms.EDAnalyzer('OuterTrackerMonitorL1Track',
    
    TopFolderName = cms.string('Phase2OuterTracker'),

# Number of TTTracks
    TH1L1Track_N = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(99.5),                      
        xmin = cms.double(-0.5)
        ),
    

)



