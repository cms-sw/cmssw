import FWCore.ParameterSet.Config as cms
import RecoTracker.TrackProducer.TrackRefitter_cfi 
refittedForPixelDQM = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    NavigationSchool = '',
    Fitter = 'FlexibleKFFittingSmoother'
)
