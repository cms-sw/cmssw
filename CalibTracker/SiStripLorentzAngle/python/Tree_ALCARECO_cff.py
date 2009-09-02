import FWCore.ParameterSet.Config as cms

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
LorentzAngleTracks = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'ALCARECOTkAlCosmicsCTF0T',  ## for ALCARECO streams
    #src = 'ctfWithMaterialTracksP5',  ## for cosmics
    #src = 'generalTracks',            ## for beam
    filter = True,
    applyBasicCuts = True,
    ptMin = 2., ##10
    ptMax = 99999.,
    etaMin = -99., ##-2.4 keep also what is going through...
    etaMax = 99., ## 2.4 ...both TEC with flat slope
    nHitMin = 7,
    nHitMin2D = 2,
    chi2nMax = 10, #999999.,
    applyNHighestPt = False, ## no pT measurement -> sort meaningless
    nHighestPt = 1,
    applyMultiplicityFilter = False
    )

from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *
LorentzAngleTracksRefit = cms.EDFilter("TrackRefitter",
                                       src = cms.InputTag("LorentzAngleTracks"),
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

oneplustracks = cms.EDFilter( "TrackCountFilter", src = cms.InputTag("LorentzAngleTracks"), minNumber = cms.uint32(1) )

trackFilterRefitter = cms.Sequence( LorentzAngleTracks + oneplustracks + offlineBeamSpot + LorentzAngleTracksRefit )

from UserCode.ShallowTools.ShallowEventDataProducer_cfi import *
from UserCode.ShallowTools.ShallowClustersProducer_cfi import *
from UserCode.ShallowTools.ShallowTrackClustersProducer_cfi import *
from UserCode.ShallowTools.ShallowTracksProducer_cfi import *
calibrationTree = cms.EDAnalyzer("ShallowTree",
                             outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_*_run_*',
    'keep *_*_clusterdetid_*',
    'keep *_*_clusterwidth_*',
    'keep *_*_clustervariance_*',
    'keep *_*_tsostrackmulti_*',
    'keep *_*_tsostrackindex_*',
    'keep *_*_tsosdriftx_*',
    'keep *_*_tsosdriftz_*',
    'keep *_*_tsoslocalpitch_*',
    'keep *_*_tsoslocaltheta_*',
    'keep *_*_tsoslocalphi_*',
    'keep *_*_tsosBdotY_*',
    'keep *_*_tsosglobalZofunitlocalY_*',
    'keep *_*_trackchi2ndof_*',
    'keep *_*_trackhitsvalid_*'
    ))
shallowTracks.Tracks = "LorentzAngleTracks"
shallowTrackClusters.Tracks = "LorentzAngleTracksRefit"

ntuple = cms.Sequence( (shallowEventRun+
                        shallowClusters +
                        shallowTracks +
                        shallowTrackClusters) *
                       calibrationTree
                       )

#Schedule
filter_refit_ntuplize_step = cms.Path( trackFilterRefitter + ntuple )
schedule = cms.Schedule( filter_refit_ntuplize_step )
