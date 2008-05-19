# The following comments couldn't be translated into the new config version:

#include "RecoEgamma/EgammaElectronProducers/data/fwdGsfElectronPropagator.cff"
# Gsf track fit, version not using Seed Association
#module pixelMatchGsfFitForGlobalGsfElectrons = GsfGlobalElectronTest from "TrackingTools/GsfTracking/data/GsfElectronFit.cfi"

import FWCore.ParameterSet.Config as cms

# $Id: globalGsfElectronSequence.cff,v 1.7 2008/04/21 09:50:46 uberthon Exp $
# create a sequence with all required modules and sources needed to make
# modules to make seeds, tracks and electrons
from RecoEgamma.EgammaElectronProducers.globalSeeds_cfi import *
# TrajectoryBuilder
#include "RecoEgamma/EgammaElectronProducers/data/gsfElectronChi2.cfi"
# "backward" propagator for electrons
from RecoEgamma.EgammaElectronProducers.bwdGsfElectronPropagator_cff import *
# "forward" propagator for electrons
from RecoEgamma.EgammaElectronProducers.fwdGsfElectronPropagator_cff import *
import RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi
TrajectoryBuilderForGlobalGsfElectrons = RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi.CkfTrajectoryBuilder.clone()
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
# Electron propagators and estimators
# Looser chi2 estimator for electron trajectory building
gsfElectronChi2ForGlobalGsfElectrons = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone()
# CKFTrackCandidateMaker
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
egammaCkfTrackCandidatesForGlobalGsfElectrons = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
# TrajectoryFilter
from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import *
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
TrajectoryFilterForGlobalGsfElectrons = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
# sources needed for GSF fit
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
import TrackingTools.GsfTracking.GsfElectronFit_cfi
pixelMatchGsfFitForGlobalGsfElectrons = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
# module to make electrons
from RecoEgamma.EgammaElectronProducers.globalGsfElectrons_cff import *
globalGsfElectronSequence = cms.Sequence(electronPixelSeedsForGlobalGsfElectrons*egammaCkfTrackCandidatesForGlobalGsfElectrons*pixelMatchGsfFitForGlobalGsfElectrons*globalGsfElectrons)
TrajectoryBuilderForGlobalGsfElectrons.ComponentName = 'TrajectoryBuilderForGlobalGsfElectrons'
TrajectoryBuilderForGlobalGsfElectrons.maxCand = 3
TrajectoryBuilderForGlobalGsfElectrons.intermediateCleaning = False
TrajectoryBuilderForGlobalGsfElectrons.propagatorAlong = 'fwdGsfElectronPropagator'
TrajectoryBuilderForGlobalGsfElectrons.propagatorOpposite = 'bwdGsfElectronPropagator'
TrajectoryBuilderForGlobalGsfElectrons.estimator = 'gsfElectronChi2ForGlobalGsfElectrons'
gsfElectronChi2ForGlobalGsfElectrons.ComponentName = 'gsfElectronChi2ForGlobalGsfElectrons'
gsfElectronChi2ForGlobalGsfElectrons.MaxChi2 = 100000.
gsfElectronChi2ForGlobalGsfElectrons.nSigma = 3.
egammaCkfTrackCandidatesForGlobalGsfElectrons.TrajectoryBuilder = 'TrajectoryBuilderForGlobalGsfElectrons'
egammaCkfTrackCandidatesForGlobalGsfElectrons.SeedProducer = 'electronPixelSeedsForGlobalGsfElectrons'
egammaCkfTrackCandidatesForGlobalGsfElectrons.SeedLabel = ''
egammaCkfTrackCandidatesForGlobalGsfElectrons.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
egammaCkfTrackCandidatesForGlobalGsfElectrons.NavigationSchool = 'SimpleNavigationSchool'
egammaCkfTrackCandidatesForGlobalGsfElectrons.RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput'
TrajectoryFilterForGlobalGsfElectrons.ComponentName = 'TrajectoryFilterForGlobalGsfElectrons'
TrajectoryFilterForGlobalGsfElectrons.filterPset = cms.PSet(
    chargeSignificance = cms.double(-1.0),
    minPt = cms.double(3.0),
    minHitsMinPt = cms.int32(-1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(1),
    maxNumberOfHits = cms.int32(-1),
    maxConsecLostHits = cms.int32(1),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)
pixelMatchGsfFitForGlobalGsfElectrons.src = 'egammaCkfTrackCandidatesForGlobalGsfElectrons'
pixelMatchGsfFitForGlobalGsfElectrons.Propagator = 'fwdGsfElectronPropagator'
pixelMatchGsfFitForGlobalGsfElectrons.Fitter = 'GsfElectronFittingSmoother'
pixelMatchGsfFitForGlobalGsfElectrons.TTRHBuilder = 'WithTrackAngle'
pixelMatchGsfFitForGlobalGsfElectrons.TrajectoryInEvent = False
pixelMatchGsfFitForGlobalGsfElectrons.producer = ''

