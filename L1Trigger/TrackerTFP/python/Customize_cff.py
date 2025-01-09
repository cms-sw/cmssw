# function to alter TrackTriggerSetup to provide TMTT configuration

import FWCore.ParameterSet.Config as cms

def setupUseTMTT(process):
  process.TrackTriggerSetup.TMTT.WidthR                   = 11
  process.TrackTriggerSetup.TMTT.WidthPhi                 = 14
  process.TrackTriggerSetup.TMTT.WidthZ                   = 13
  process.TrackTriggerSetup.TrackFinding.MinPt            =  3.0
  process.TrackTriggerSetup.TrackFinding.MaxEta           =  2.4
  process.TrackTriggerSetup.TrackFinding.ChosenRofPhi     = 67.24
  process.TrackTriggerSetup.TrackFinding.NumLayers        =  8
  process.TrackTriggerSetup.GeometricProcessor.ChosenRofZ = 57.76
  process.TrackTriggerSetup.HoughTransform.MinLayers      =  5
  process.TrackTriggerSetup.CleanTrackBuilder.MaxStubs    =  4
  process.TrackTriggerSetup.KalmanFilter.NumWorker        =  4
  process.TrackTriggerSetup.KalmanFilter.MaxLayers        =  8
  process.TrackTriggerSetup.KalmanFilter.MaxSeedingLayer  =  3
  process.TrackTriggerSetup.KalmanFilter.MaxGaps          =  2
  process.TrackTriggerSetup.KalmanFilter.ShiftChi20       =  0
  process.TrackTriggerSetup.KalmanFilter.ShiftChi21       =  0
  process.TrackTriggerSetup.KalmanFilter.CutChi2          =  2.0

def simUseTMTT(process):
  process.StubAssociator.MinPt =  3.0

  return process
