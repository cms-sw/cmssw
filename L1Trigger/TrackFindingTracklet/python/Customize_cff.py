# functions to alter configurations

import FWCore.ParameterSet.Config as cms

# configures track finding s/w to behave as track finding f/w
def fwConfig(process):
  process.l1tTTTracksFromTrackletEmulation.Fakefit = True
  process.TrackTriggerSetup.TrackFinding.MaxEta =  2.5
  process.TrackTriggerSetup.GeometricProcessor.ChosenRofZ = 57.76
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
  process.ProducerKF.Hybrid                                   = True
  process.ProducerKF.DeadModuleOpts.KillScenario              = 0
  process.ProducerKF.DeadModuleOpts.KillRecover               = False
  process.ProducerKF.HTArraySpecRphi.HoughMinPt               = 2.
  process.ProducerKF.TrackDigi.KF_skipTrackDigi               = True
  process.ProducerKF.StubDigitize.EnableDigitize              = False
  process.ProducerKF.GeometricProc.UseApproxB                 = True
  process.ProducerKF.GeometricProc.BApprox_gradient           = 0.886454
  process.ProducerKF.GeometricProc.BApprox_intercept          = 0.504148
  process.ProducerKF.PhiSectors.NumPhiSectors                 = 9
  process.ProducerKF.PhiSectors.NumPhiNonants                 = 9
  process.ProducerKF.PhiSectors.ChosenRofPhi                  = 55.
  process.ProducerKF.EtaSectors.EtaRegions                    = [-2.4, -2.08, -1.68, -1.26, -0.90, -0.62, -0.41, -0.20, 0.0, 0.20, 0.41, 0.62, 0.90, 1.26, 1.68, 2.08, 2.4]
  process.ProducerKF.EtaSectors.ChosenRofZ                    = 50.0
  process.ProducerKF.TrackFitSettings.KalmanMinNumStubs       = 4
  process.ProducerKF.TrackFitSettings.KalmanMaxNumStubs       = 6
  process.ProducerKF.TrackFitSettings.KalmanMaxSkipLayersHard = 1
  process.ProducerKF.TrackFitSettings.KalmanMaxSkipLayersEasy = 2
  process.ProducerKF.TrackFitSettings.KalmanMaxStubsEasy      = 10
  process.ProducerKF.TrackFitSettings.KalmanMaxStubsPerLayer  = 4
  process.ProducerKF.TrackFitSettings.KalmanMultiScattTerm    = 0.00075
  process.ProducerKF.TrackFitSettings.KalmanChi2RphiScale     = 8
  process.ProducerKF.TrackFitSettings.KFUseMaybeLayers        = True
  process.ProducerKF.TrackFitSettings.KalmanRemove2PScut      = True
  process.ProducerKF.TrackFitSettings.KFLayerVsPtToler        = [999., 999., 0.1, 0.1, 0.05, 0.05, 0.05]
  process.ProducerKF.TrackFitSettings.KFLayerVsD0Cut5         = [999., 999., 999., 10., 10., 10., 10.]
  process.ProducerKF.TrackFitSettings.KFLayerVsZ0Cut5         = [999., 999., 25.5, 25.5, 25.5, 25.5, 25.5]
  process.ProducerKF.TrackFitSettings.KFLayerVsZ0Cut4         = [999., 999., 15., 15., 15., 15., 15.]
  process.ProducerKF.TrackFitSettings.KFLayerVsChiSq5         = [999., 999., 10., 30., 80., 120., 160.]
  process.ProducerKF.TrackFitSettings.KFLayerVsChiSq4         = [999., 999., 10., 30., 80., 120., 160.]
  process.ProducerKF.TrackFitSettings.KalmanAddBeamConstr     = False
  process.ProducerKF.TrackFitSettings.KalmanHOfw              = False
  process.ProducerKF.TrackFitSettings.KalmanHOtilted          = True
  process.ProducerKF.TrackFitSettings.KalmanHOprojZcorr       = 1
  process.ProducerKF.TrackFitSettings.KalmanHOalpha           = 0
  process.ProducerKF.TrackFitSettings.KalmanHOhelixExp        = True
  process.ProducerKF.TrackFitSettings.KalmanDebugLevel        = 0