import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripLorentzAngle.Tree_RECO_cff import *

from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_cff import *
ALCARECOTkAlCosmicsCTF0T.src='ALCARECOTkAlCosmicsCTF0T'
#ALCARECOTkAlCosmicsCTF0T.ptMin=5.

from RecoTracker.TrackProducer.TrackRefitters_cff import *
trackRefitter = cms.EDFilter("TrackRefitter",
                             src = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"),
                             beamSpot = cms.InputTag("offlineBeamSpot"),
                             constraint = cms.string(''),
                             srcConstr  = cms.InputTag(''),
                             Fitter = cms.string('RKFittingSmoother'),
                             useHitsSplitting = cms.bool(False),
                             TrajectoryInEvent = cms.bool(True),
                             TTRHBuilder = cms.string('WithTrackAngle'),
                             AlgorithmName = cms.string('ctf'),
                             Propagator = cms.string('RungeKuttaTrackerPropagator')
                             )
refit = cms.Sequence(seqALCARECOTkAlCosmicsCTF0T+offlineBeamSpot+trackRefitter)

oneplustracks = cms.EDFilter( "TrackCountFilter", src = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"), minNumber = cms.uint32(1) )

shallowTracks.Tracks = "ALCARECOTkAlCosmicsCTF0T"
shallowTrackClusters.Tracks = "trackRefitter"

#Schedule
refit_step = cms.Path( refit )
ntuple_step = cms.Path( oneplustracks+theBigNtuple )
schedule = cms.Schedule( refit_step , ntuple_step)
