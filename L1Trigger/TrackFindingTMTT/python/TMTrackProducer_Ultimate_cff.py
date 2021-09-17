import FWCore.ParameterSet.Config as cms

#---------------------------------------------------------------------------------------------------------
# This describes the full TMTT track reconstruction chain with 2 GeV threshold, where:
# the GP divides the tracker into 18 eta sectors (each sub-divided into 2 virtual eta subsectors);  
# the HT uses a  32x24 array followed by 2x2 Mini-HT array, with transverese HT readout & multiplexing, 
# followed by the track fit (KF); duplicate track removal (Algo1) is run.
#
# It represents the tracking as planned for 2026. It is a good basis for L1 trigger studies etc.
#---------------------------------------------------------------------------------------------------------

#=== TMTT tracking needs to get FE stub window sizes from this.

from L1Trigger.TrackTrigger.TTStubAlgorithmRegister_cfi import *

#=== Random number generator for Stub Killer (dead module emulation)

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    TMTrackProducer = cms.PSet(initialSeed = cms.untracked.uint32(12345))
)

#=== Import default values for all parameters & define EDProducer.

from L1Trigger.TrackFindingTMTT.TMTrackProducer_Defaults_cfi import TMTrackProducer_params

TMTrackProducer = cms.EDProducer('tmtt::TMTrackProducer',
  # Load cfg parameters from TMTrackProducer_Defaults_cfi.py
  TMTrackProducer_params
)

#===================================================================================================
# Uncomment the following 2 lines to enable use of MC truth info & output histograms.
# (This costs CPU, and is unnecessary if you only care about producing TTTrack collection).
#===================================================================================================

#TMTrackProducer.EnableMCtruth = True
#TMTrackProducer.EnableHistos  = True

#===================================================================================================
#=== The following override the default values.
#===================================================================================================

#--- Configure track fitter(s).

# Use only 4 parameter helix fit Kalman Filter.
TMTrackProducer.TrackFitSettings.TrackFitters = ["KF4ParamsComb"]

# Allow KF to assign stubs in up to this many layers to fitted tracks.
TMTrackProducer.TrackFitSettings.KalmanMaxNumStubs  = 6
# Enable more sophisticated fit mathematics in KF.
TMTrackProducer.TrackFitSettings.KalmanHOtilted     = True
TMTrackProducer.TrackFitSettings.KalmanHOhelixExp   = True
TMTrackProducer.TrackFitSettings.KalmanHOalpha      = 1
TMTrackProducer.TrackFitSettings.KalmanHOprojZcorr  = 1
TMTrackProducer.TrackFitSettings.KalmanHOfw      = False
TMTrackProducer.TrackFitSettings.KFUseMaybeLayers   = True

#--- Switch on 2nd stage Mini HT with 2 GeV Pt threshold & allow it to find tracks with stubs in as few as 4 layers.

TMTrackProducer.HTArraySpecRphi.HoughNbinsPt        = 48
TMTrackProducer.HTArraySpecRphi.HoughNbinsPhi       = 64 
TMTrackProducer.GenCuts.GenMinPt                    = 2.0
TMTrackProducer.HTArraySpecRphi.HoughMinPt          = 2.0
TMTrackProducer.HTArraySpecRphi.MiniHoughMinPt      = 3.0 # Mini-HT not used below this Pt, to reduce sensitivity to scattering.
TMTrackProducer.L1TrackDef.MinStubLayers            = 4 
TMTrackProducer.HTFillingRphi.BusySectorMbinRanges  = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 24]
TMTrackProducer.HTFillingRphi.BusySectorMbinOrder   = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47]

#--- phi digitisation range needs to be increased to go down to 2 GeV.
#--- phi0 digitised relative to centre of sector. (Required range 2pi/18 + 2*overlap; overlap = 0.19206rads*(2GeV/ptCut)*(chosenR/67.24)

TMTrackProducer.TrackDigi.KF_phi0Range = 2*0.6981317
# FIX: To make this change in KF FW, change phi0 bit selection in DRstate.vhd to bits 17-6 (instead of 16-5).

# MuxHToutputs sends all HT outputs for an entire nonant and 1 m-bin to a single output link.
# This works for Pt > 3 GeV, gives truncation for Pt > 2 GeV. To solve, need to double number of outputs,
# with one for each phi sector in nonant. Not yet implemented, so for now disable HT output truncation.
TMTrackProducer.HTFillingRphi.BusySectorNumStubs = 999 
