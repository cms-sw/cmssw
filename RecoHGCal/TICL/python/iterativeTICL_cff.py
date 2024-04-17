import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.FastJetStep_cff import *
from RecoHGCal.TICL.CLUE3DHighStep_cff import *
from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.TrkEMStep_cff import *
from RecoHGCal.TICL.TrkStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *
from RecoHGCal.TICL.CLUE3DEM_cff import *
from RecoHGCal.TICL.CLUE3DHAD_cff import *

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer
from RecoHGCal.TICL.pfTICLProducerV5_cfi import pfTICLProducerV5 as _pfTICLProducerV5
from RecoHGCal.TICL.trackstersMergeProducer_cfi import trackstersMergeProducer as _trackstersMergeProducer
from RecoHGCal.TICL.tracksterSelectionTf_cfi import *

from RecoHGCal.TICL.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoHGCal.TICL.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer

from RecoHGCal.TICL.mtdSoAProducer_cfi import mtdSoAProducer as _mtdSoAProducer

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

ticlTrackstersMerge = _trackstersMergeProducer.clone()
ticlTracksterLinks = _tracksterLinksProducer.clone()
ticlCandidate = _ticlCandidateProducer.clone()
mtdSoA = _mtdSoAProducer.clone()

pfTICL = _pfTICLProducer.clone()
ticl_v5.toModify(pfTICL, ticlCandidateSrc = cms.InputTag('ticlCandidate'), isTICLv5 = cms.bool(True))

ticlPFTask = cms.Task(pfTICL)

ticlIterationsTask = cms.Task(
    ticlCLUE3DHighStepTask,
    ticlCLUE3DEMStepTask,
    ticlCLUE3DHADStepTask
)

ticl_v5.toReplaceWith(ticlIterationsTask, ticlIterationsTask.copyAndExclude([ticlCLUE3DHighStepTask]))

from Configuration.ProcessModifiers.fastJetTICL_cff import fastJetTICL
fastJetTICL.toModify(ticlIterationsTask, func=lambda x : x.add(ticlFastJetStepTask))

ticlIterLabels = ["CLUE3DEM", "CLUE3DHAD", "CLUE3DHigh"]

ticlTracksterMergeTask = cms.Task(ticlTrackstersMerge)
ticlTracksterLinksTask = cms.Task(ticlTracksterLinks)

mergeTICLTask = cms.Task(ticlLayerTileTask
    ,ticlIterationsTask
    ,ticlTracksterMergeTask
    ,ticlTracksterLinksTask # needed for the customizer
)
ticl_v5.toReplaceWith(mergeTICLTask, mergeTICLTask.copyAndExclude([ticlTracksterMergeTask]))
ticl_v5.toModify(mergeTICLTask, func=lambda x : x.add(ticlTracksterLinksTask))

ticlIterLabelsMerge = ticlIterLabels + ["Merge"]

mtdSoATask = cms.Task(mtdSoA)
ticlCandidateTask = cms.Task(ticlCandidate)

iterTICLTask = cms.Task(mergeTICLTask,
    mtdSoATask, ticlCandidateTask, # needed for the customizer
    ticlPFTask)
ticl_v5.toModify(iterTICLTask, func=lambda x : x.add(mtdSoATask, ticlCandidateTask))

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
