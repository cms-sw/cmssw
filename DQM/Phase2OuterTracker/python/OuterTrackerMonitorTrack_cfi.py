import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorTrack = cms.EDAnalyzer('OuterTrackerMonitorTrack',
    
    TopFolderName  = cms.string('Phase2OuterTracker'),
    TTTracks       = cms.InputTag("TTTracksFromPixelDigis", "Level1TTTracks"),
    HQDelim        = cms.int32(4),


# Number of Stubs
    TH1_NStubs = cms.PSet(
        Nbinsx = cms.int32(11),
        xmax = cms.double(10.5),                      
        xmin = cms.double(-0.5)
        ),

# Number of TTTracks
    TH1_NTracks = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(99.5),                      
        xmin = cms.double(-0.5)
        ),

#Pt of the track
    TH1_Track_Pt = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(100),                      
        xmin = cms.double(0)
        ),

#Phi of the track
    TH1_Track_Phi = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416)
        ),

#Eta of the track
    TH1_Track_Eta = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3.0),                      
        xmin = cms.double(-3.0)
        ),

#VtxZ0 of the track
    TH1_Track_VtxZ0 = cms.PSet(
        Nbinsx = cms.int32(51),
        xmax = cms.double(25),                      
        xmin = cms.double(-25)
        ),

#Chi2 of the track
    TH1_Track_Chi2 = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(50),                      
        xmin = cms.double(0)
        ),

#Chi2R of the track
    TH1_Track_Chi2R = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(10),                      
        xmin = cms.double(0)
        ),

#Chi2 of the track vs Nb of stubs
    TH2_Track_Chi2_NStubs = cms.PSet(
        Nbinsx = cms.int32(11),
        xmax = cms.double(10.5),                      
        xmin = cms.double(-0.5), 
        Nbinsy = cms.int32(100),
        ymax = cms.double(50),                      
        ymin = cms.double(0)
        ),

#Chi2R of the track vs Nb of stubs
    TH2_Track_Chi2R_NStubs = cms.PSet(
        Nbinsx = cms.int32(11),
        xmax = cms.double(10.5),                      
        xmin = cms.double(-0.5), 
        Nbinsy = cms.int32(100),
        ymax = cms.double(10),                      
        ymin = cms.double(0)
        ),
	
	
	
)



