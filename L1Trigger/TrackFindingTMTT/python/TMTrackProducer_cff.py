import FWCore.ParameterSet.Config as cms

#---------------------------------------------------------------------------------------------------------
# This describes the full TMTT track reconstruction chain with 3 GeV threshold, where:
# the GP divides the tracker into 18 eta sectors (each sub-divided into 2 virtual eta subsectors);  
# the HT uses a  32x18 array followed by 2x2 Mini-HT array, with transverese HT readout & multiplexing, 
# followed by the KF (or optionally SF+SLR) track fit; duplicate track removal (Algo50) is run.
#
# This usually corresponds to the current firmware.
#---------------------------------------------------------------------------------------------------------

#=== Import default values for all parameters & define EDProducer.

from L1Trigger.TrackFindingTMTT.TMTrackProducer_Defaults_cfi import TMTrackProducer_params

TMTrackProducer = cms.EDProducer('TMTrackProducer',
  # Load cfg parameters from TMTrackProducer_Defaults_cfi.py
  TMTrackProducer_params
)

#===================================================================================================
#=== Parameters changed from their default values.
#===================================================================================================

#--- Disable internal digitisation of SimpleLR fitter, as it was never retuned for nonants.
TMTrackProducer.TrackFitSettings.DigitizeSLR = cms.bool(False)

#===================================================================================================
#=== All the following parameters already have identical values in TMTrackProducer_Defaults_cfi .
#=== They are listed here just to remind you of the most interesting parameters to play with.
#===================================================================================================

#--- Configure track fitting

# Use only 4 or 5 parameter helix fit Kalman Filter (which automatically runs on tracks produced with no r-z track filter)
#TMTrackProducer.TrackFitSettings.TrackFitters = cms.vstring("KF5ParamsComb", "KF4ParamsComb")
# Use only Linear Regression Fitter (which automatically runs on tracks produced by r-z track filter).
#TMTrackProducer.TrackFitSettings.TrackFitters = cms.vstring("SimpleLR")

# Allow KF to assign stubs in up to this many layers to fitted tracks.
#TMTrackProducer.TrackFitSettings.KalmanMaxNumStubs  = cms.uint32(6)
# Enable more sophisticated fit mathematics in KF.
#TMTrackProducer.TrackFitSettings.KalmanHOtilted     = cms.bool(True)
#TMTrackProducer.TrackFitSettings.KalmanHOhelixExp   = cms.bool(True)
#TMTrackProducer.TrackFitSettings.KalmanHOalpha      = cms.uint32(2)
#TMTrackProducer.TrackFitSettings.KalmanHOprojZcorr  = cms.uint32(2)
#TMTrackProducer.TrackFitSettings.KalmanHOdodgy      = cms.bool(False)

#--- Switch off parts of the track reconstruction chain.

#TMTrackProducer.DupTrkRemoval.DupTrkAlgRphi   = cms.uint32(0)
#TMTrackProducer.DupTrkRemoval.DupTrkAlg3D     = cms.uint32(0)
#TMTrackProducer.DupTrkRemoval.DupTrkAlgFit    = cms.uint32(0)
#TMTrackProducer.TrackFitSettings.TrackFitters = cms.vstring()

#--- Keep Pt threshold at 3 GeV, with coarse HT, but switch off Mini-HT.

#TMTrackProducer.HTArraySpecRphi.MiniHTstage         = cms.bool(False)  
#TMTrackProducer.HTFillingRphi.MaxStubsInCell        = cms.uint32(16) 
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPt        = cms.uint32(16) 
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPhi       = cms.uint32(32) 
#TMTrackProducer.HTFillingRphi.BusySectorMbinRanges  = cms.vuint32(2,2,2,2,2,2,2,2) 
#TMTrackProducer.HTFillingRphi.BusySectorMbinOrder   = cms.vuint32(0,8, 1,9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15)

#--- Reduce Pt threshold to 2 GeV, with coarse HT, and switch off Mini-HT.

#TMTrackProducer.HTArraySpecRphi.MiniHTstage        = cms.bool(False)  
#TMTrackProducer.HTFillingRphi.MaxStubsInCell       = cms.uint32(16) 
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPt       = cms.uint32(24)
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPhi      = cms.uint32(32) 
#TMTrackProducer.GenCuts.GenMinPt                   = cms.double(2.0)
#TMTrackProducer.HTArraySpecRphi.HoughMinPt         = cms.double(2.0)
#TMTrackProducer.HTFillingRphi.BusySectorMbinRanges = cms.vuint32(2,2,2,2,2,2,2,2,2,2,2,2)   
#TMTrackProducer.HTFillingRphi.BusySectorMbinOrder  = cms.vuint32(0,12, 1,13, 2,14, 3,15, 4,16, 5,17, 6,18, 7,19, 8,20, 9,21, 10,22, 11,23)

#--- Reduce Pt threshold to 2 GeV, with coarse HT, followed  by Mini-HT.

#TMTrackProducer.HTArraySpecRphi.HoughNbinsPt        = cms.uint32(48)
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPhi       = cms.uint32(64) 
#TMTrackProducer.GenCuts.GenMinPt                    = cms.double(2.0)
#TMTrackProducer.HTArraySpecRphi.HoughMinPt          = cms.double(2.0)
#TMTrackProducer.HTArraySpecRphi.MiniHoughMinPt      = cms.double(3.0) # Mini-HT not used below this Pt, to reduce sensitivity to scattering.
#TMTrackProducer.HTFillingRphi.BusySectorMbinRanges  = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 24)
#TMTrackProducer.HTFillingRphi.BusySectorMbinOrder   = cms.vuint32(0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47)

#--- Additional Mini-HT options to improve electron/displaced tracking.

# Next 2 lines cause tracks found by 1st stage HT to be output if above specified Pt threshold & mini-HT found no tracks.
# Improves electron tracking. Setting Pt threshold to 0 improves displaced tracking.
#TMTrackProducer.HTArraySpecRphi.MiniHoughDontKill   = cms.bool(True)  
#TMTrackProducer.HTArraySpecRphi.MiniHoughDontKillMinPt   = cms.double(8.)  
# Extreme displaced tracking also benefits from following.
#TMTrackProducer.L1TrackDef.MinStubLayers            = cms.uint32(4)  # HT accepts tracks with >= 4 layers
#TMTrackProducer.TrackFitSettings.KalmanRemove2PScut = cms.bool(True)
#To study displaced tracking, include non-prompt particles in efficiency definition.
#TMTrackProducer.GenCuts.GenMaxVertR                 = cms.double(30.) 

#--- Unusual HT cell shapes

# Simplify HT MUX to allow easy playing with the number of m bins.
#TMTrackProducer.HTFillingRphi.BusySectorMbinOrder  = cms.vuint32() 

# Diamond shaped cells: (64,62), (34,32) or (46,44) sized array interesting.
#TMTrackProducer.HTArraySpecRphi.Shape         = cms.uint32(1)
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPt  = cms.uint32(38)  
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPhi = cms.uint32(32)  

# Hexagonal shaped cells: (64,42), (50,32) or (56,36) sized array interesting.
#TMTrackProducer.HTArraySpecRphi.Shape         = cms.uint32(2)
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPt  = cms.uint32(56)  
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPhi = cms.uint32(32)  

# Brick-wall arranged cells: (64,30) or (66,32) sized array interesting.
#TMTrackProducer.HTArraySpecRphi.Shape         = cms.uint32(3)
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPt  = cms.uint32(64)  
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPhi = cms.uint32(27)  

#--- Older cfg giving similar tracking performance with slightly larger resource use.

#TMTrackProducer.PhiSectors.NumPhiSectors           = cms.uint32(36)
#TMTrackProducer.EtaSectors.EtaRegions = cms.vdouble(-2.4, -2.0, -1.53, -0.98, -0.37, 0.37, 0.98, 1.53, 2.0, 2.4)
#TMTrackProducer.EtaSectors.ChosenRofZ              = cms.double(45.)     
#TMTrackProducer.EtaSectors.AllowOver2EtaSecs       = cms.bool(False)
#TMTrackProducer.HTArraySpecRphi.HoughNbinsPhi      = cms.uint32(32)
#TMTrackProducer.HTArraySpecRphi.NumSubSecsEta      = cms.uint32(1)

#--- Stub digitization (switch on/off and/or change defaults).

#TMTrackProducer.StubDigitize.EnableDigitize  = cms.bool(True)

#--- Reduce requirement on number of layers a track must have stubs in, either globally or in specific eta regions.

#TMTrackProducer.L1TrackDef.MinStubLayers       = cms.uint32(4)  # Reduce it globally
#TMTrackProducer.L1TrackDef.EtaSecsReduceLayers = cms.vuint32(5,12) # barrel-endcap transition region

#--- If globally reducing number of layers cut, best to also use just one HT output opto-link per m-bin.
# For 3 GeV threshold with no mini-HT.
#TMTrackProducer.HTFillingRphi.BusySectorMbinRanges = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)   
# For 2 GeV threshold with mini-HT.
#TMTrackProducer.HTFillingRphi.BusySectorMbinRanges = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 24)   

#--- Change TP to track matching criteria.

#TMTrackProducer.GenCuts.GenMinStubLayers               = cms.uint32(4)
#TMTrackProducer.TrackMatchDef.MinNumMatchLayers        = cms.uint32(4)

#--- Switch off data truncation due to finite band-width.

#TMTrackProducer.HTFillingRphi.BusySectorKill       = cms.bool(False)
#TMTrackProducer.HTFillingRphi.BusyInputSectorKill  = cms.bool(False)

# Don't order stubs by bend in DTC, such that highest Pt stubs are transmitted first.
#TMTrackProducer.StubCuts.OrderStubsByBend = cms.bool(False)

#--- Switch on FPGA-friendly approximation to B parameter in GP - will be used in future GP firmware.
#--- (used to relate track angle dphi to stub bend) 
#TMTrackProducer.GeometricProc.UseApproxB           = cms.bool(True)
