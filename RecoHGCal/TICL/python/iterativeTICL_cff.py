import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.TrkStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.ticlCandidateFromTrackstersProducer_cfi import ticlCandidateFromTrackstersProducer as _ticlCandidateFromTrackstersProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer
from RecoHGCal.TICL.trackstersMergeProducer_cfi import trackstersMergeProducer as _trackstersMergeProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

ticlTrackstersMerge = _trackstersMergeProducer.clone()
ticlMultiClustersFromTrackstersMerge = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlTrackstersMerge"
)
ticlTracksterMergeTask = cms.Task(ticlTrackstersMerge, ticlMultiClustersFromTrackstersMerge)

ticlCandidateFromTracksters = _ticlCandidateFromTrackstersProducer.clone(
      tracksterCollections = ["ticlTrackstersMerge"],
      # A possible alternative for momentum computation:
      # momentumPlugin = dict(plugin="TracksterP4FromTrackAndPCA")
    )
pfTICL = _pfTICLProducer.clone()
ticlPFTask = cms.Task(ticlCandidateFromTracksters, pfTICL)

iterTICLTask = cms.Task(ticlLayerTileTask
    ,ticlMIPStepTask
    ,ticlTrkStepTask
    ,ticlEMStepTask
    ,ticlHADStepTask
    ,ticlTracksterMergeTask
    ,ticlPFTask
    )

####

ticlLayerTileHFNoseProducer = ticlLayerTileProducer.clone(
    detector = 'HFNose'
)
_hfnose_ticlLayerTileTask = cms.Task(ticlLayerTileHFNoseProducer)

_hfnose_iterTICLTask = iterTICLTask.copy()
_hfnose_iterTICLTask.add(_hfnose_ticlLayerTileTask)

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
phase2_hfnose.toReplaceWith(
    iterTICLTask, _hfnose_iterTICLTask )

####

def injectTICLintoPF(process):
    if getattr(process,'particleFlowTmp', None):
      process.particleFlowTmp.src = ['particleFlowTmpBarrel', 'pfTICL']

    return process
