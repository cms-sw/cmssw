import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorL1Track = cms.EDAnalyzer('OuterTrackerMonitorL1Track',
    
    TopFolderName = cms.string('Phase2OuterTracker'),

# Number of Stubs
    TH1_NStubs = cms.PSet(
        Nbinsx = cms.int32(16),
        xmax = cms.double(15.5),                      
        xmin = cms.double(-0.5)
        ),
	
	
# Number of TTTracks
    TH1_NL1Tracks = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(99.5),                      
        xmin = cms.double(-0.5)
        ),

# Nb of stubs vs phi sector or eta wedge
    TH2_NStubs_PhiSectorOrEtaWedge = cms.PSet(
        Nbinsx = cms.int32(35),
        xmax = cms.double(34.5),                      
        xmin = cms.double(-0.5),
	Nbinsy = cms.int32(20), 
	ymax = cms.double(19.5),
	ymin = cms.double(-0.5)
        ),
	
# Sector or wedge vs resp Phi or eta track
    TH2_PhiSectorOrEtaWedge_PhiOrEta = cms.PSet(
        Nbinsx = cms.int32(200),
        xmax = cms.double(4),                      
        xmin = cms.double(-4),
	Nbinsy = cms.int32(35), 
	ymax = cms.double(34.5),
	ymin = cms.double(-0.5)
        ),

#Pt of the track
    TH1_L1Track_Pt = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(100),                      
        xmin = cms.double(0)
        ),
#Phi of the track
    TH1_L1Track_Phi = cms.PSet(
        Nbinsx = cms.int32(90),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416)
        ),
#Eta of the track
    TH1_L1Track_Eta = cms.PSet(
        Nbinsx = cms.int32(90),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416)
        ),


#VtxZ0 of the track
    TH1_L1Track_VtxZ0 = cms.PSet(
        Nbinsx = cms.int32(150),
        xmax = cms.double(25),                      
        xmin = cms.double(-25)
        ),

#Chi2 of the track
    TH1_L1Track_Chi2 = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(50),                      
        xmin = cms.double(0)
        ),

#Chi2R of the track
    TH1_L1Track_Chi2R = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(10),                      
        xmin = cms.double(0)
        ),

#Chi2 of the track vs Nb of stubs
    TH2_L1Track_Chi2_NStubs = cms.PSet(
        Nbinsx = cms.int32(20),
        xmax = cms.double(19.5),                      
        xmin = cms.double(-0.5), 
	Nbinsy = cms.int32(100),
        ymax = cms.double(50),                      
        ymin = cms.double(0)
        ),

#Chi2R of the track vs Nb of stubs
    TH2_L1Track_Chi2R_NStubs = cms.PSet(
        Nbinsx = cms.int32(20),
        xmax = cms.double(19.5),                      
        xmin = cms.double(-0.5), 
	Nbinsy = cms.int32(100),
        ymax = cms.double(10),                      
        ymin = cms.double(0)
        ),
	
	
	
)



