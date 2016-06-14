###############################################################
# 
# Configuration blocks for the TrajectoryFactories inheriting 
# from TrajectoryFactoryBase.
# Include this file and do e.g.
# TrajectoryFactory = cms.PSet( ReferenceTrajectoryFactory)
# 
###############################################################

import FWCore.ParameterSet.Config as cms

###############################################################
#
# Common to all TrajectoryFactories
#
###############################################################
__muonMass = cms.double(0.10565836)

TrajectoryFactoryBase = cms.PSet(
    PropagationDirection = cms.string('alongMomentum'), ## or "oppositeToMomentum" or "anyDirection"
    MaterialEffects = cms.string('Combined'), ## or "MultipleScattering" or "EnergyLoss" or "None"
                                              ## (see others at 'BrokenLinesTrajectoryFactory')
    UseProjectedHits = cms.bool(True), ## if false, projected hits are skipped
    UseInvalidHits = cms.bool(False), ## if false, invalid hits are skipped
    UseHitWithoutDet = cms.bool(True), ## if false, RecHits that are not attached to GeomDets are skipped
    UseBeamSpot = cms.bool(False), ## if true, the beam spot is used as a constraint via a virtual TTRecHit
    IncludeAPEs = cms.bool(False) ## if true, the APEs are included in the hit error
)

###############################################################
#
# ReferenceTrajectoryFactory
#
###############################################################
ReferenceTrajectoryFactory = cms.PSet(
    TrajectoryFactoryBase,
    ParticleMass = __muonMass,
    TrajectoryFactoryName = cms.string('ReferenceTrajectoryFactory'),
    UseBzeroIfFieldOff = cms.bool(True), # if true, use BzeroReferenceTrajectory if B == 0
    MomentumEstimateFieldOff = cms.double(10.) # used if useBzeroIfFieldOff == True

)

###############################################################
#
# BzeroReferenceTrajectoryFactory
#
###############################################################
BzeroReferenceTrajectoryFactory = cms.PSet(
    TrajectoryFactoryBase,
    ParticleMass = __muonMass,
    TrajectoryFactoryName = cms.string('BzeroReferenceTrajectoryFactory'),
    MomentumEstimate = cms.double(10.0)
)

###############################################################
#
# DualTrajectoryFactory
#
###############################################################
DualTrajectoryFactory = cms.PSet(
    TrajectoryFactoryBase,
    ParticleMass = __muonMass,
    TrajectoryFactoryName = cms.string('DualTrajectoryFactory')
)

###############################################################
#
# DualBzeroTrajectoryFactory
#
###############################################################
DualBzeroTrajectoryFactory = cms.PSet(
    TrajectoryFactoryBase,
    ParticleMass = __muonMass,
    TrajectoryFactoryName = cms.string('DualBzeroTrajectoryFactory'),
    MomentumEstimate = cms.double(10.0)
)

###############################################################
#
# TwoBodyDecayReferenceTrajectoryFactory
#
###############################################################
TwoBodyDecayTrajectoryFactory = cms.PSet(
    TrajectoryFactoryBase,
    NSigmaCut = cms.double(100.0),
    Chi2Cut = cms.double(10000.0),
    ParticleProperties = cms.PSet(
        PrimaryMass = cms.double(91.1876),
        PrimaryWidth = cms.double(2.4952),
        SecondaryMass = cms.double(0.105658)
    ),
    ConstructTsosWithErrors = cms.bool(False),
    UseRefittedState = cms.bool(True),
    EstimatorParameters = cms.PSet(
        MaxIterationDifference = cms.untracked.double(0.01),
        RobustificationConstant = cms.untracked.double(1.0),
        MaxIterations = cms.untracked.int32(100),
        UseInvariantMass = cms.untracked.bool(True)
    ),
    TrajectoryFactoryName = cms.string('TwoBodyDecayTrajectoryFactory')
)

###############################################################
#
# CombinedTrajectoryFactory using an instance of TwoBodyDecayTrajectoryFactory
# and ReferenceTrajectoryFactory, taking the first successful.
#
###############################################################
CombinedTrajectoryFactory = cms.PSet(
    TrajectoryFactoryBase, # will not be used!
    TrajectoryFactoryName = cms.string('CombinedTrajectoryFactory'),
    # look for PSets called TwoBody and Reference:
    TrajectoryFactoryNames = cms.vstring(
        'TwoBodyDecayTrajectoryFactory,TwoBody',  # look for PSet called TwoBody
        'ReferenceTrajectoryFactory,Reference'),  # look for PSet called Reference
    useAllFactories = cms.bool(False),
    # now one PSet for each of the configured trajectories:
    TwoBody = cms.PSet( # FIXME: better by reference?
        TwoBodyDecayTrajectoryFactory
    ),
    Reference = cms.PSet( # FIXME: better by reference?
        ReferenceTrajectoryFactory
    )
)
###############################################################
#
# CombinedTrajectoryFactory using two instances of BzeroReferenceTrajectoryFactory,
# one propagating alongMomentum, one oppositeToMomentum.
#
###############################################################
# First a helper object, where I'd like to do:
#BwdBzeroReferenceTrajectoryFactory = BzeroReferenceTrajectoryFactory.clone(PropagationDirection = 'oppositeToMomentum')
# Since there is no clone in cms.PSet (yet?), but clone is needed for python that works by reference, 
# take solution from https://hypernews.cern.ch/HyperNews/CMS/get/swDevelopment/1890/1.html:
import copy
BwdBzeroReferenceTrajectoryFactory = copy.deepcopy(BzeroReferenceTrajectoryFactory)
BwdBzeroReferenceTrajectoryFactory.PropagationDirection = 'oppositeToMomentum'
# now the PSet
CombinedFwdBwdBzeroTrajectoryFactory = cms.PSet(
    TrajectoryFactoryBase, # will not be used!
    TrajectoryFactoryName = cms.string('CombinedTrajectoryFactory'),

    TrajectoryFactoryNames = cms.vstring(
        'BzeroReferenceTrajectoryFactory,FwdBzero',  # look for PSet called FwdBzero
        'BzeroReferenceTrajectoryFactory,BwdBzero'), # look for PSet called BwdBzero
    useAllFactories = cms.bool(True),
    
    # now one PSet for each of the configured trajectories:
    FwdBzero = cms.PSet(BzeroReferenceTrajectoryFactory), # FIXME: better by reference?
    BwdBzero = cms.PSet(BwdBzeroReferenceTrajectoryFactory) # FIXME: better by reference?
)

###############################################################
#
# CombinedTrajectoryFactory using three ReferenceTrajectories:
# - two instances of BzeroReferenceTrajectoryFactory,
#   one propagating alongMomentum, one oppositeToMomentum,
# - a DualBzeroTrajectory to start in the middle.
#
###############################################################
CombinedFwdBwdDualBzeroTrajectoryFactory = cms.PSet(
    TrajectoryFactoryBase, # will not be used!
    TrajectoryFactoryName = cms.string('CombinedTrajectoryFactory'),

    TrajectoryFactoryNames = cms.vstring(
        'BzeroReferenceTrajectoryFactory,FwdBzero',  # look for PSet called FwdBzero
        'BzeroReferenceTrajectoryFactory,BwdBzero',  # look for PSet called BwdBzero
    	'DualBzeroTrajectoryFactory,DualBzero'),     # look for PSet called DualBzero
    useAllFactories = cms.bool(True),

    # now one PSet for each of the configured trajectories:
    FwdBzero  = cms.PSet(BzeroReferenceTrajectoryFactory), # FIXME: better by reference?
    BwdBzero  = cms.PSet(BwdBzeroReferenceTrajectoryFactory), # defined above for CombinedFwdBwdBzeroTrajectoryFactory  # FIXME: better by reference?
    DualBzero = cms.PSet(DualBzeroTrajectoryFactory) # FIXME: better by reference?
)


###############################################################
#
# CombinedTrajectoryFactory using two instances of ReferenceTrajectoryFactory,
# one propagating alongMomentum, one oppositeToMomentum.
#
###############################################################
# First a helper object, see above for CombinedFwdBwdBzeroTrajectoryFactory:
BwdReferenceTrajectoryFactory = copy.deepcopy(ReferenceTrajectoryFactory)
BwdReferenceTrajectoryFactory.PropagationDirection = 'oppositeToMomentum'
# now the PSet
CombinedFwdBwdTrajectoryFactory = cms.PSet(
    TrajectoryFactoryBase, # will not be used!
    TrajectoryFactoryName = cms.string('CombinedTrajectoryFactory'),

    TrajectoryFactoryNames = cms.vstring(
        'ReferenceTrajectoryFactory,Fwd',  # look for PSet called Fwd
        'ReferenceTrajectoryFactory,Bwd'), # look for PSet called Bwd
    useAllFactories = cms.bool(True),
    
    # now one PSet for each of the configured trajectories:
    Fwd = cms.PSet(ReferenceTrajectoryFactory), # FIXME: better by reference?
    Bwd = cms.PSet(BwdReferenceTrajectoryFactory)  # FIXME: better by reference?
)

###############################################################
#
# CombinedTrajectoryFactory using three ReferenceTrajectories:
# - two instances of ReferenceTrajectoryFactory,
#   one propagating alongMomentum, one oppositeToMomentum,
# - a DualTrajectory to start in the middle.
#
###############################################################
CombinedFwdBwdDualTrajectoryFactory = cms.PSet(
    TrajectoryFactoryBase, # will not be used!
    TrajectoryFactoryName = cms.string('CombinedTrajectoryFactory'),

    TrajectoryFactoryNames = cms.vstring(
        'ReferenceTrajectoryFactory,Fwd',  # look for PSet called Fwd
        'ReferenceTrajectoryFactory,Bwd',  # look for PSet called Bwd
    	'DualTrajectoryFactory,Dual'),     # look for PSet called Dual
    useAllFactories = cms.bool(True),

    # now one PSet for each of the configured trajectories:
    Fwd  = cms.PSet(ReferenceTrajectoryFactory),  # FIXME: better by reference?
    Bwd  = cms.PSet(BwdReferenceTrajectoryFactory), # defined above for CombinedFwdBwdTrajectoryFactory # FIXME: better by reference?
    Dual = cms.PSet(DualTrajectoryFactory) # FIXME: better by reference?
)

###############################################################
#
# ReferenceTrajectoryFactory with BrokenLines
#
###############################################################
BrokenLinesTrajectoryFactory = ReferenceTrajectoryFactory.clone(
    MaterialEffects = 'BrokenLinesCoarse', # same as "BrokenLines"
              # others are "BrokenLinesCoarsePca" == "BrokenLinesPca",
              #            "BrokenLinesFine", "BrokenLinesFinePca"
              #             or even "BreakPoints"
    UseInvalidHits = True # to account for multiple scattering in these layers
    )


###############################################################
#
# BzeroReferenceTrajectoryFactory with BrokenLines
#
###############################################################
BrokenLinesBzeroTrajectoryFactory = BzeroReferenceTrajectoryFactory.clone(
    MaterialEffects = 'BrokenLinesCoarse', # see BrokenLinesTrajectoryFactory
    UseInvalidHits = True # to account for multiple scattering in these layers
    )
