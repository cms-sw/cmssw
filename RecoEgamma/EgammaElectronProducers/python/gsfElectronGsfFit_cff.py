import FWCore.ParameterSet.Config as cms

# Gsf track fit for GsfElectrons
from RecoEgamma.EgammaElectronProducers.fwdGsfElectronPropagator_cff import *
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
import TrackingTools.GsfTracking.GsfElectronFit_cfi
pixelMatchGsfFit = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
pixelMatchGsfFit.src = 'egammaCkfTrackCandidates'
pixelMatchGsfFit.Propagator = 'fwdGsfElectronPropagator'
pixelMatchGsfFit.Fitter = 'GsfElectronFittingSmoother'
pixelMatchGsfFit.TTRHBuilder = 'WithTrackAngle'
pixelMatchGsfFit.TrajectoryInEvent = True

