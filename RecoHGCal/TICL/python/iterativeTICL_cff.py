import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.TrkStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.ticlCandidateFromTrackstersProducer_cfi import ticlCandidateFromTrackstersProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer
from RecoHGCal.TICL.trackstersMergeProducer_cfi import trackstersMergeProducer as _trackstersMergeProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer


ticlLayerTileTask = cms.Task(ticlLayerTileProducer)
ticlPFTask = cms.Task(ticlCandidateFromTrackstersProducer, pfTICLProducer)
trackstersMerge = _trackstersMergeProducer.clone()
multiClustersFromTrackstersMerge = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "trackstersMerge"
)
ticlTracksterMergeTask = cms.Task(trackstersMerge, multiClustersFromTrackstersMerge)

iterTICLTask = cms.Task(ticlLayerTileTask
    ,MIPStepTask
    ,EMStepTask
    ,TrkStepTask
    ,HADStepTask
    ,ticlTracksterMergeTask
    ,ticlPFTask
    )

iterTICL = cms.Sequence(iterTICLTask)

