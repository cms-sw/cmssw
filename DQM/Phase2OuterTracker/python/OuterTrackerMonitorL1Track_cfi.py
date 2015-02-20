import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorL1Track = cms.EDAnalyzer('OuterTrackerMonitorL1Track',
    
    TopFolderName = cms.string('Phase2OuterTracker'),

# Number of TTTracks
    TH1L1Track_N = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(99.5),                      
        xmin = cms.double(-0.5)
        ),

# Nb of stubs vs phi sector or eta wedge
    TH2L1Track_N_PhiSectorOrEtaWedge = cms.PSet(
        Nbinsx = cms.int32(35),
        xmax = cms.double(34.5),                      
        xmin = cms.double(-0.5),
	Nbinsy = cms.int32(20), 
	ymax = cms.double(19.5),
	ymin = cms.double(-0.5)
        ),
	
# Sector or wedge vs resp Phi or eta track
    TH2L1Track_PhiOrEta = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(3.14),                      
        xmin = cms.double(-3.14),
	Nbinsy = cms.int32(35), 
	ymax = cms.double(34.5),
	ymin = cms.double(-0.5)
        ),

#Pt of the track
    TH1L1Track_Pt = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(50),                      
        xmin = cms.double(0)
        ),
#Phi of the track
    TH1L1Track_Phi = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(3.14),                      
        xmin = cms.double(-3.14)
        ),
#Eta of the track
    TH1L1Track_Eta = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(3.14),                      
        xmin = cms.double(-3.14)
        ),
	
#Theta of the track
    TH1L1Track_Theta = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(6.28),                      
        xmin = cms.double(-6.28)
        ),

#VtxZ0 of the track
    TH1L1Track_VtxZ0 = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(5),                      
        xmin = cms.double(-5)
        ),

#Chi2 of the track
    TH1L1Track_Chi2 = cms.PSet(
        Nbinsx = cms.int32(200),
        xmax = cms.double(50),                      
        xmin = cms.double(0)
        ),

#Chi2R of the track
    TH1L1Track_Chi2R = cms.PSet(
        Nbinsx = cms.int32(200),
        xmax = cms.double(10),                      
        xmin = cms.double(0)
        ),

#Chi2 of the track vs Nb of stubs
    TH2L1Track_Chi2_N = cms.PSet(
        Nbinsx = cms.int32(20),
        xmax = cms.double(19.5),                      
        xmin = cms.double(-0.5), 
	Nbinsy = cms.int32(200),
        ymax = cms.double(50),                      
        ymin = cms.double(0)
        ),

#Chi2R of the track vs Nb of stubs
    TH2L1Track_Chi2R_N = cms.PSet(
        Nbinsx = cms.int32(20),
        xmax = cms.double(19.5),                      
        xmin = cms.double(-0.5), 
	Nbinsy = cms.int32(200),
        ymax = cms.double(10),                      
        ymin = cms.double(0)
        ),
	
	
	
)



