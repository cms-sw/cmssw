# function to alter TrackTriggerSetup to provide TMTT configuration

import FWCore.ParameterSet.Config as cms

def setupUseTMTT(process):
  process.TrackTriggerSetup.TMTT.WidthR                   = 11       # number of bits used for stub r - ChosenRofPhi
  process.TrackTriggerSetup.TMTT.WidthPhi                 = 14       # number of bits used for stub phi w.r.t. phi region centre
  process.TrackTriggerSetup.TMTT.WidthZ                   = 13       # number of bits used for stub z
  process.TrackTriggerSetup.TrackFinding.MinPt            =  3.0     # min track pt in GeV, also defines region overlap shape
  process.TrackTriggerSetup.TrackFinding.MaxEta           =  2.4     # cut on stub eta
  process.TrackTriggerSetup.TrackFinding.ChosenRofPhi     = 67.24    # critical radius defining region overlap shape in cm
  process.TrackTriggerSetup.TrackFinding.NumLayers        =  8       # TMTT: number of detector layers a reconstructbale particle may cross, reduced to 7, 8th layer almost never corssed
  process.TrackTriggerSetup.GeometricProcessor.ChosenRofZ = 57.76    # critical radius defining r-z sector shape in cm
  process.TrackTriggerSetup.HoughTransform.MinLayers      =  5       # required number of stub layers to form a candidate
  process.TrackTriggerSetup.CleanTrackBuilder.MaxStubs    =  4       # cut on number of stub per layer for input candidates
  process.TrackTriggerSetup.KalmanFilter.NumWorker        =  4       # number of kf worker
  process.TrackTriggerSetup.KalmanFilter.MaxLayers        =  8       # maximum number of  layers added to a track
  process.TrackTriggerSetup.KalmanFilter.MaxSeedingLayer  =  3       # perform seeding in layers 0 to this
  process.TrackTriggerSetup.KalmanFilter.MaxGaps          =  2       # maximum number of layer gaps allowed during cominatorical track building
  process.TrackTriggerSetup.KalmanFilter.ShiftChi20       =  0       # shiting chi2 in r-phi plane by power of two when caliclating chi2
  process.TrackTriggerSetup.KalmanFilter.ShiftChi21       =  0       # shiting chi2 in r-z plane by power of two when caliclating chi2
  process.TrackTriggerSetup.KalmanFilter.CutChi2          =  2.0     # cut on chi2 over degree of freedom

def simUseTMTT(process):
  process.StubAssociator.MinPt =  3.0 # pt cut in GeV

  return process
