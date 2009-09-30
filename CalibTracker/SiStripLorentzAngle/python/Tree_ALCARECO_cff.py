import FWCore.ParameterSet.Config as cms

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
LorentzAngleTracks = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'ALCARECOTkAlCosmicsCTF0T',  ## for ALCARECO streams
    #src = 'ctfWithMaterialTracksP5',  ## for cosmics
    #src = 'generalTracks',            ## for beam
    filter = True,
    applyBasicCuts = True,
    ptMin = 0.5,
    nHitMin = 4,
    chi2nMax = 10.,
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

trackFilterRefitter = cms.Sequence( LorentzAngleTracks + offlineBeamSpot + LorentzAngleTracksRefit )

from CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowClustersProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowTrackClustersProducer_cfi import *
calibrationTree = cms.EDAnalyzer("ShallowTree",
                                 outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_*_clusterdetid_*',
    'keep *_*_clusterwidth_*',
    'keep *_*_clustervariance_*',
    'keep *_*_tsostrackmulti_*',
    'keep *_*_tsosdriftx_*',
    'keep *_*_tsosdriftz_*',
    'keep *_*_tsoslocalpitch_*',
    'keep *_*_tsoslocaltheta_*',
    'keep *_*_tsoslocalphi_*',
    'keep *_*_tsosBdotY_*',
    'keep *_*_tsosglobalZofunitlocalY_*'
    ))
shallowTrackClusters.Tracks = "LorentzAngleTracksRefit"
shallowTrackClusters.Clusters = 'LorentzAngleTracks'
shallowClusters.Clusters = 'LorentzAngleTracks'

ntuple = cms.Sequence( (shallowEventRun+
                        shallowClusters +
                        shallowTrackClusters) *
                       calibrationTree
                       )

#Schedule
filter_refit_ntuplize_step = cms.Path( trackFilterRefitter + ntuple )
schedule = cms.Schedule( filter_refit_ntuplize_step )
