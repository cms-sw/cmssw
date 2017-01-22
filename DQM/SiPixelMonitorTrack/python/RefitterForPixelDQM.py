import FWCore.ParameterSet.Config as cms
import RecoTracker.TrackProducer.TrackRefitter_cfi 
refittedForPixelDQM = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
refittedForPixelDQM.NavigationSchool = ''
refittedForPixelDQM.Fitter = 'FlexibleKFFittingSmoother'
