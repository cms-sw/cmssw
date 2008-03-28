import FWCore.ParameterSet.Config as cms

# Gsf track fit for GsfElectrons
from RecoEgamma.EgammaElectronProducers.fwdGsfElectronPropagator_cff import *
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
import copy
from TrackingTools.GsfTracking.GsfElectronFit_cfi import *
pixelMatchGsfFit = copy.deepcopy(GsfGlobalElectronTest)
pixelMatchGsfFit.src = 'egammaCkfTrackCandidates'
pixelMatchGsfFit.Propagator = 'fwdGsfElectronPropagator'
pixelMatchGsfFit.Fitter = 'GsfElectronFittingSmoother'
pixelMatchGsfFit.TTRHBuilder = 'WithTrackAngle'
pixelMatchGsfFit.TrajectoryInEvent = False

