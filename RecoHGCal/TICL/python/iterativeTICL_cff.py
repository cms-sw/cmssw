import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.FastJetStep_cff import *
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
from RecoHGCal.TICL.trackstersMergeProducerV3_cfi import trackstersMergeProducerV3 as _trackstersMergeProducerV3
from RecoHGCal.TICL.tracksterSelectionTf_cfi import *

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

ticlTrackstersMerge = _trackstersMergeProducer.clone()
ticlTrackstersMergeV3 = _trackstersMergeProducerV3.clone()

pfTICL = _pfTICLProducer.clone()
ticlPFTask = cms.Task(pfTICL)

ticlIterationsTask = cms.Task(
    ticlCLUE3DHighStepTask
)

from Configuration.ProcessModifiers.clue3D_cff import clue3D
clue3D.toModify(ticlIterationsTask, func=lambda x : x.add(ticlCLUE3DHighStepTask,ticlCLUE3DLowStepTask))

from Configuration.ProcessModifiers.fastJetTICL_cff import fastJetTICL
fastJetTICL.toModify(ticlIterationsTask, func=lambda x : x.add(ticlFastJetStepTask))

from Configuration.ProcessModifiers.ticl_v3_cff import ticl_v3
ticl_v3.toModify(ticlIterationsTask, func=lambda x : x.add( ticlTrkEMStepTask
    ,ticlEMStepTask
    ,ticlTrkStepTask
    ,ticlHADStepTask) )
ticlIterLabels = [_step.itername.value() for _iteration in ticlIterationsTask for _step in _iteration if (_step._TypedParameterizable__type == "TrackstersProducer")]

ticlTracksterMergeTask = cms.Task(ticlTrackstersMerge)
ticlTracksterMergeTaskV3 = cms.Task(ticlTrackstersMergeV3)

ticl_v3.toModify(pfTICL, ticlCandidateSrc = "ticlTrackstersMergeV3")

mergeTICLTask = cms.Task(ticlLayerTileTask
    ,ticlIterationsTask
    ,ticlTracksterMergeTask
)

ticl_v3.toModify(mergeTICLTask, func=lambda x : x.add(ticlTracksterMergeTaskV3))
ticlIterLabelsMerge = ticlIterLabels + ["Merge"]

ticlIterLabelsMergeV3 = ticlIterLabels + ["MergeV3"]
ticl_v3.toModify(ticlIterLabelsMerge, func=lambda x : x.extend(ticlIterLabelsMergeV3))

iterTICLTask = cms.Task(mergeTICLTask
    ,ticlPFTask)

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
