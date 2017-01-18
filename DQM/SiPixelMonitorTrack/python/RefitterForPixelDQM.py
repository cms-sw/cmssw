import FWCore.ParameterSet.Config as cms
import RecoTracker.TrackProducer.TrackRefitter_cfi 
refittedForPixelDQM = RecoTracker.TrackProducer.TrackRefitter_cfiTrackRefitter.clone()
process.RefittedForPixelDQM.NavigationSchool = ''
process.RefittedForPixelDQM.Fitter = 'FlexibleKFFittingSmoother'
