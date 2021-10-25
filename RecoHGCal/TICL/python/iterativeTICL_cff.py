import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.CLUE3DHighStep_cff import *
from RecoHGCal.TICL.CLUE3DLowStep_cff import *
from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.TrkEMStep_cff import *
from RecoHGCal.TICL.TrkStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer
from RecoHGCal.TICL.trackstersMergeProducer_cfi import trackstersMergeProducer as _trackstersMergeProducer

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

ticlTrackstersMerge = _trackstersMergeProducer.clone()
ticlTracksterMergeTask = cms.Task(ticlTrackstersMerge)


pfTICL = _pfTICLProducer.clone()
ticlPFTask = cms.Task(pfTICL)

ticlIterationsTask = cms.Task(
    ticlTrkEMStepTask
    ,ticlEMStepTask
    ,ticlTrkStepTask
    ,ticlHADStepTask
)

from Configuration.ProcessModifiers.clue3D_cff import clue3D
clue3D.toModify(ticlIterationsTask, func=lambda x : x.add(ticlCLUE3DHighStepTask,ticlCLUE3DLowStepTask))

ticlIterLabels = [_step.itername.value() for _iteration in ticlIterationsTask for _step in _iteration if (_step._TypedParameterizable__type == "TrackstersProducer")]

iterTICLTask = cms.Task(ticlLayerTileTask
    ,ticlIterationsTask
    ,ticlTracksterMergeTask
    ,ticlPFTask
)
ticlIterLabelsMerge = ticlIterLabels + ["Merge"]

ticlLayerTileHFNose = ticlLayerTileProducer.clone(
    detector = 'HFNose'
)

ticlLayerTileHFNoseTask = cms.Task(ticlLayerTileHFNose)

iterHFNoseTICLTask = cms.Task(ticlLayerTileHFNoseTask
    ,ticlHFNoseTrkEMStepTask
    ,ticlHFNoseEMStepTask
    ,ticlHFNoseTrkStepTask
    ,ticlHFNoseHADStepTask
    ,ticlHFNoseMIPStepTask
)
