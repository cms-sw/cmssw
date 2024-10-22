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
from RecoHGCal.TICL.PRbyRecovery_cff import *

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer
from RecoHGCal.TICL.trackstersMergeProducer_cfi import trackstersMergeProducer as _trackstersMergeProducer
from RecoHGCal.TICL.tracksterSelectionTf_cfi import *

from RecoHGCal.TICL.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoHGCal.TICL.superclustering_cff import *
from RecoHGCal.TICL.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer

from RecoHGCal.TICL.mtdSoAProducer_cfi import mtdSoAProducer as _mtdSoAProducer

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
from Configuration.ProcessModifiers.ticl_superclustering_dnn_cff import ticl_superclustering_dnn
from Configuration.ProcessModifiers.ticl_superclustering_mustache_pf_cff import ticl_superclustering_mustache_pf
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

ticlTrackstersMerge = _trackstersMergeProducer.clone()
ticlTracksterLinks = _tracksterLinksProducer.clone(
    tracksters_collections = cms.VInputTag(
        'ticlTrackstersCLUE3DHigh',
        'ticlTrackstersRecovery'
    ),
    regressionAndPid = cms.bool(True),
    inferenceAlgo = cms.string('TracksterInferenceByDNN'),
    pluginInferenceAlgoTracksterInferenceByDNN = cms.PSet(
        algo_verbosity = cms.int32(0),
        doPID = cms.int32(1),
        doRegression = cms.int32(1),
        inputNames  = cms.vstring('input'),
        output_en   = cms.vstring('enreg_output'),
	output_id   = cms.vstring('pid_output'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_clusters = cms.int32(10),
        eid_n_layers = cms.int32(50),
        onnxEnergyModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/linking/energy_v0.onnx'),
        onnxPIDModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/linking/id_v0.onnx'),
        type = cms.string('TracksterInferenceByDNN')
    )
)

ticlCandidate = _ticlCandidateProducer.clone()
mtdSoA = _mtdSoAProducer.clone()

pfTICL = _pfTICLProducer.clone()
ticl_v5.toModify(pfTICL, ticlCandidateSrc = cms.InputTag('ticlCandidate'), isTICLv5 = cms.bool(True), useTimingAverage=True)

ticlPFTask = cms.Task(pfTICL)

ticlIterationsTask = cms.Task(
    ticlCLUE3DHighStepTask
)

ticl_v5.toModify(ticlIterationsTask , func=lambda x : x.add(ticlRecoveryStepTask))
''' For future separate iterations
,ticlCLUE3DEMStepTask,
,ticlCLUE3DHADStepTask
    '''

''' For future separate iterations
ticl_v5.toReplaceWith(ticlIterationsTask, ticlIterationsTask.copyAndExclude([ticlCLUE3DHighStepTask]))
'''

from Configuration.ProcessModifiers.fastJetTICL_cff import fastJetTICL
fastJetTICL.toModify(ticlIterationsTask, func=lambda x : x.add(ticlFastJetStepTask))

ticlIterLabels = ["ticlTrackstersCLUE3DHigh", "ticlTrackstersMerge"]
ticlIterLabels_v5 = ["ticlTrackstersCLUE3DHigh", "ticlTracksterLinks", "ticlCandidate"]

''' For future separate iterations
"CLUE3DEM", "CLUE3DHAD",
'''

ticlTracksterMergeTask = cms.Task(ticlTrackstersMerge)
ticlTracksterLinksTask = cms.Task(ticlTracksterLinks, ticlSuperclusteringTask) 


mergeTICLTask = cms.Task(ticlLayerTileTask
    ,ticlIterationsTask
    ,ticlTracksterMergeTask
)

ticl_v5.toReplaceWith(mergeTICLTask, mergeTICLTask.copyAndExclude([ticlTracksterMergeTask]))
ticl_v5.toModify(mergeTICLTask, func=lambda x : x.add(ticlTracksterLinksTask))


mtdSoATask = cms.Task(mtdSoA)
ticlCandidateTask = cms.Task(ticlCandidate)


if ticl_v5._isChosen():
    ticlIterLabels = ticlIterLabels_v5.copy()
    if ticl_superclustering_mustache_ticl._isChosen():
        ticlIterLabels.append("ticlTracksterLinksSuperclusteringMustache")
    if ticl_superclustering_dnn._isChosen():
        ticlIterLabels.append("ticlTracksterLinksSuperclusteringDNN")


associatorsInstances = []

for labelts in ticlIterLabels:
    for labelsts in ['ticlSimTracksters', 'ticlSimTrackstersfromCPs']:
        associatorsInstances.append(labelts+'To'+labelsts)
        associatorsInstances.append(labelsts+'To'+labelts)

iterTICLTask = cms.Task(mergeTICLTask,
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
