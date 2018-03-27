import FWCore.ParameterSet.Config as cms
import math

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
OuterTrackerMonitorTTTrack = DQMEDAnalyzer('OuterTrackerMonitorTTTrack',
    TopFolderName  = cms.string('SiOuterTracker'),
    TTTracksTag    = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"), #tracks (currently from tracklet)
    HQNStubs       = cms.int32(5),  #cut for "high quality" tracks
    HQChi2dof      = cms.double(10.0), #cut for "high quality" tracks

# Number of stubs per track
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
        Nbinsx = cms.int32(60),
        xmax = cms.double(math.pi),
        xmin = cms.double(-math.pi)
        ),

#D0 of the track
    TH1_Track_D0 = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(25),
        xmin = cms.double(-25)
        ),

#Eta of the track
    TH1_Track_Eta = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3.0),
        xmin = cms.double(-3.0)
        ),

#VtxZ of the track
    TH1_Track_VtxZ = cms.PSet(
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

#Chi probability of the track
    TH1_Track_Chi2_Probability = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(1),
        xmin = cms.double(0)
        ),

#Chi2R of the track vs Nb of stubs
    TH2_Track_Chi2R_NStubs = cms.PSet(
        Nbinsx = cms.int32(11),
        xmax = cms.double(10.5),
        xmin = cms.double(-0.5),
        Nbinsy = cms.int32(15),
        ymax = cms.double(10),
        ymin = cms.double(0)
        ),

#Chi2R of the track vs eta
    TH2_Track_Chi2R_Eta = cms.PSet(
        Nbinsx = cms.int32(15),
        xmax = cms.double(3.0),
        xmin = cms.double(-3.0),
        Nbinsy = cms.int32(15),
        ymax = cms.double(10),
        ymin = cms.double(0)
        ),

#Eta of the track vs Nb of stubs (in barrel, in EC, and total)
    TH2_Track_Eta_NStubs = cms.PSet(
        Nbinsx = cms.int32(15),
        xmax = cms.double(3.0),
        xmin = cms.double(-3.0),
        Nbinsy = cms.int32(11),
        ymax = cms.double(10.5),
        ymin = cms.double(-0.5)
        ),
)
