# functions to alter configurations

import FWCore.ParameterSet.Config as cms

# configures track finding s/w to behave as track finding f/w
def fwConfig(process):
  process.l1tTTTracksFromTrackletEmulation.Fakefit = True
  process.l1tTTTracksFromTrackletEmulation.RemovalType = ""
  process.l1tTTTracksFromTrackletEmulation.DoMultipleMatches = False
  process.l1tTTTracksFromTrackletEmulation.StoreTrackBuilderOutput = True

# configures track finding s/w to behave as a subchain of processing steps
def reducedConfig(process):
  fwConfig(process)
  process.TrackTriggerSetup.Firmware.FreqBEHigh = 240 # Frequency of DTC & KF (determines truncation)
  process.TrackTriggerSetup.KalmanFilter.NumWorker = 1
  process.ChannelAssignment.SeedTypes = cms.vstring( "L1L2" )
  process.ChannelAssignment.SeedTypesSeedLayers = cms.PSet( L1L2 = cms.vint32( 1,  2 ) )
  process.ChannelAssignment.SeedTypesProjectionLayers = cms.PSet( L1L2 = cms.vint32(  3,  4,  5,  6 ) )
  # this are tt::Setup::dtcId in order as in process.l1tTTTracksFromTrackletEmulation.processingModulesFile translated by 
  # reverssing naming logic described in L1FPGATrackProducer
  # TO DO: Eliminate cfg param IRChannelsIn by taking this info from Tracklet wiring map.
  process.ChannelAssignment.IRChannelsIn = cms.vint32( 0, 1, 25, 2, 26, 4, 5, 29, 6, 30, 7, 31, 8, 9, 33 )
  process.l1tTTTracksFromTrackletEmulation.Reduced = True
  process.l1tTTTracksFromTrackletEmulation.memoryModulesFile = 'L1Trigger/TrackFindingTracklet/data/reduced_memorymodules.dat'
  process.l1tTTTracksFromTrackletEmulation.processingModulesFile = 'L1Trigger/TrackFindingTracklet/data/reduced_processingmodules.dat'
  process.l1tTTTracksFromTrackletEmulation.wiresFile = 'L1Trigger/TrackFindingTracklet/data/reduced_wires.dat'

# configures pure tracklet algorithm (as opposed to Hybrid algorithm)
def trackletConfig(process):
  process.l1tTTTracksFromTrackletEmulation.fitPatternFile = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/fitpattern.txt') 

# configures KF simulation in emulation chain
def oldKFConfig(process):
  #===== Use HYBRID TRACKING (Tracklet pattern reco + TMTT KF -- requires tracklet C++ too) =====
  process.ProducerKF.Hybrid                                   = True
  # Emulate dead/inefficient modules using the StubKiller code, with stubs killed according to the scenarios of the Stress Test group. 
  # (0=Don't kill any stubs; 1-5 = Scenarios described in StubKiller.cc) 
  process.ProducerKF.DeadModuleOpts.KillScenario              = 0
  # Modify TMTT tracking to try to recover tracking efficiency in presence of dead modules. (Does nothing if KillScenario = 0).
  process.ProducerKF.DeadModuleOpts.KillRecover               = False
  # Min track Pt that Hough Transform must find. Also used by StubCuts.KillLowPtStubs and by EtaPhiSectors.UseStubPhi.
  process.ProducerKF.HTArraySpecRphi.HoughMinPt               = 2.
  # Optionally skip track digitisation if done internally inside fitting code.
  process.ProducerKF.TrackDigi.KF_skipTrackDigi               = True
  # Digitize stub coords? If not, use floating point coords.
  process.ProducerKF.StubDigitize.EnableDigitize              = False
  # Use an FPGA-friendly approximation to determine track angle dphi from bend in GP?
  process.ProducerKF.GeometricProc.UseApproxB                 = True
  # Gradient term of linear equation for approximating B
  process.ProducerKF.GeometricProc.BApprox_gradient           = 0.886454
  # Intercept term of linear equation for approximating B
  process.ProducerKF.GeometricProc.BApprox_intercept          = 0.504148
  # Divisions of Tracker at GP.
  process.ProducerKF.PhiSectors.NumPhiSectors                 = 9
  # Divisions of Tracker at DTC
  process.ProducerKF.PhiSectors.NumPhiNonants                 = 9
  # Use phi of track at this radius for assignment of stubs to phi sectors & also for one of the axes of the r-phi HT. If ChosenRofPhi=0, then use track phi0. - Should be an integer multiple of the stub r digitisation granularity.
  process.ProducerKF.PhiSectors.ChosenRofPhi                  = 55.
  # Eta boundaries for 16 eta regions
  process.ProducerKF.EtaSectors.EtaRegions                    = [-2.4, -2.08, -1.68, -1.26, -0.90, -0.62, -0.41, -0.20, 0.0, 0.20, 0.41, 0.62, 0.90, 1.26, 1.68, 2.08, 2.4]
  # Use z of track at this radius for assignment of tracks to eta sectors & also for one of the axes of the r-z HT. Do not set to zero!
  process.ProducerKF.EtaSectors.ChosenRofZ                    = 50.0
  # Fit will reject fitted tracks unless it can assign at least this number of stubs to them.
  process.ProducerKF.TrackFitSettings.KalmanMinNumStubs       = 4
  # Fit will attempt to add up to this nummber of stubs to each fitted tracks, but won't bother adding more.
  process.ProducerKF.TrackFitSettings.KalmanMaxNumStubs       = 6
  # Allow the KF to skip this many layers in total per track.
  process.ProducerKF.TrackFitSettings.KalmanMaxSkipLayersHard = 1
  # For HT tracks with few stubs
  process.ProducerKF.TrackFitSettings.KalmanMaxSkipLayersEasy = 2
   # Max stubs an HT track can have to be "easy".
  process.ProducerKF.TrackFitSettings.KalmanMaxStubsEasy      = 10
  # KF will consider at most this #stubs per layer to save time.
  process.ProducerKF.TrackFitSettings.KalmanMaxStubsPerLayer  = 4
  # Multiple scattering term - inflate hit phi errors by this divided by Pt (0.00075 gives best helix resolution & 0.00450 gives best chi2 distribution).
  process.ProducerKF.TrackFitSettings.KalmanMultiScattTerm    = 0.00075
  # Scale down chi2 in r-phi plane by this factor to improve electron performance (should be power of 2)
  process.ProducerKF.TrackFitSettings.KalmanChi2RphiScale     = 8
  # Disable "maybe layer" to match with firmware
  process.ProducerKF.TrackFitSettings.KFUseMaybeLayers        = True
  # Remove requirement of at least 2 PS layers per track.
  process.ProducerKF.TrackFitSettings.KalmanRemove2PScut      = True
  #--- Cuts applied to KF states as a function of the last KF tracker layer they had a stub in.
  # (If "4" or "5" in name, cut only applies to 4 or 5 param helix fit).
  process.ProducerKF.TrackFitSettings.KFLayerVsPtToler        = [999., 999., 0.1, 0.1, 0.05, 0.05, 0.05]
  # d0 cut only applied to 5 param helix fit.
  process.ProducerKF.TrackFitSettings.KFLayerVsD0Cut5         = [999., 999., 999., 10., 10., 10., 10.]
  process.ProducerKF.TrackFitSettings.KFLayerVsZ0Cut5         = [999., 999., 25.5, 25.5, 25.5, 25.5, 25.5]
  process.ProducerKF.TrackFitSettings.KFLayerVsZ0Cut4         = [999., 999., 15., 15., 15., 15., 15.]
  # Chi2 cuts should be retuned if KalmanMultiScattTerm value changed.
  process.ProducerKF.TrackFitSettings.KFLayerVsChiSq5         = [999., 999., 10., 30., 80., 120., 160.]
  process.ProducerKF.TrackFitSettings.KFLayerVsChiSq4         = [999., 999., 10., 30., 80., 120., 160.]
  # For 5-param helix fits, calculate also beam-constrained helix params after fit is complete, & use them for duplicate removal if DupTrkAlgFit=1.
  process.ProducerKF.TrackFitSettings.KalmanAddBeamConstr     = False
  # Use approx calc to account for non-radial endcap 2S modules corresponding to current FW, with  no special treatment for tilted modules.
  process.ProducerKF.TrackFitSettings.KalmanHOfw              = False
  # Treat z uncertainty in tilted barrel modules correctly.
  process.ProducerKF.TrackFitSettings.KalmanHOtilted          = True
  # Projection from (r,phi) to (z,phi) for endcap 2S modules. (0=disable correction, 1=correct with offset, 2=correct with non-diagonal stub covariance matrix). -- Option 1 is easier in FPGA, but only works if fit adds PS stubs before 2S ones.
  process.ProducerKF.TrackFitSettings.KalmanHOprojZcorr       = 1
  # Alpha correction for non-radial 2S endcap strips. (0=disable correction, 1=correct with offset, 2=correct with non-diagonal stub covariance matrix). -- Option 1 is easier in FPGA, but only works if fit adds PS stubs before 2S ones.
  process.ProducerKF.TrackFitSettings.KalmanHOalpha           = 0
  # Higher order circle explansion terms for low Pt.
  process.ProducerKF.TrackFitSettings.KalmanHOhelixExp        = True
  # Larger number has more debug printout. "1" is useful for understanding why tracks are lost, best combined with TrackFitCheat=True.
  process.ProducerKF.TrackFitSettings.KalmanDebugLevel        = 0
