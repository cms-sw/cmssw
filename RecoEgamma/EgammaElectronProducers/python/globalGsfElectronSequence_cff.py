import FWCore.ParameterSet.Config as cms

# $Id: globalGsfElectronSequence_cff.py,v 1.2 2008/04/21 03:24:53 rpw Exp $
# create a sequence with all required modules and sources needed to make
# modules to make seeds, tracks and electrons
from RecoEgamma.EgammaElectronProducers.electronPixelSeeds_cff import *
import copy
from RecoEgamma.EgammaElectronProducers.electronPixelSeeds_cfi import *
electronPixelSeedsForGlobalGsfElectrons = copy.deepcopy(electronPixelSeeds)
# TrajectoryBuilder
#include "RecoEgamma/EgammaElectronProducers/data/gsfElectronChi2.cfi"
# "backward" propagator for electrons
from RecoEgamma.EgammaElectronProducers.bwdGsfElectronPropagator_cff import *

from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
TrajectoryBuilderForGlobalGsfElectrons = copy.deepcopy(CkfTrajectoryBuilder)

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# Electron propagators and estimators
# Looser chi2 estimator for electron trajectory building
gsfElectronChi2ForGlobalGsfElectrons = copy.deepcopy(Chi2MeasurementEstimator)
# CKFTrackCandidateMaker
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *

from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
egammaCkfTrackCandidatesForGlobalGsfElectrons = copy.deepcopy(ckfTrackCandidates)
# TrajectoryFilter
from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import *

from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
TrajectoryFilterForGlobalGsfElectrons = copy.deepcopy(trajectoryFilterESProducer)
# sources needed for GSF fit
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
#from RecoEgamma.EgammaElectronProducers.fwdGsfElectronPropagator_cff import *

from TrackingTools.GsfTracking.GsfElectronFit_cfi import *
# Gsf track fit, version not using Seed Association
pixelMatchGsfFitForGlobalGsfElectrons = copy.deepcopy(GsfGlobalElectronTest)
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

