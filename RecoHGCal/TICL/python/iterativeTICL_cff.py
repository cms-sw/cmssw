import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.TrkEMStep_cff import *
from RecoHGCal.TICL.TrkStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer
from RecoHGCal.TICL.trackstersMergeProducer_cfi import trackstersMergeProducer as _trackstersMergeProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer as _multiClustersFromTrackstersProducer

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

ticlTrackstersMerge = _trackstersMergeProducer.clone()
ticlMultiClustersFromTrackstersMerge = _multiClustersFromTrackstersProducer.clone(
    Tracksters = "ticlTrackstersMerge"
)
ticlTracksterMergeTask = cms.Task(ticlTrackstersMerge, ticlMultiClustersFromTrackstersMerge)


pfTICL = _pfTICLProducer.clone()
ticlPFTask = cms.Task(pfTICL)

iterTICLTask = cms.Task(ticlLayerTileTask
    ,ticlTrkEMStepTask
    ,ticlEMStepTask
    ,ticlTrkStepTask
    ,ticlHADStepTask
    ,ticlTracksterMergeTask
    ,ticlPFTask
    )

ticlLayerTileHFNose = ticlLayerTileProducer.clone(
    detector = 'HFNose'
)

ticlLayerTileHFNoseTask = cms.Task(ticlLayerTileHFNose)

iterHFNoseTICLTask = cms.Task(
    ticlLayerTileHFNoseTask,
    ticlHFNoseMIPStepTask,
    ticlHFNoseEMStepTask
)

def _findIndicesByModule(process,name):
    ret = []
    if getattr(process,'particleFlowBlock', None):
        for i, pset in enumerate(process.particleFlowBlock.elementImporters):
            if pset.importerName.value() == name:
                ret.append(i)
    return ret

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
def injectTICLintoPF(process):
    if getattr(process,'particleFlowTmp', None):
      process.particleFlowTmp.src = ['particleFlowTmpBarrel', 'pfTICL']

    _insertTrackImportersWithVeto = {}
    _trackImporters = ['GeneralTracksImporter','ConvBremTrackImporter',
                   'ConversionTrackImporter','NuclearInteractionTrackImporter']
    for importer in _trackImporters:
        for idx in _findIndicesByModule(process,importer):
            _insertTrackImportersWithVeto[idx] = dict(
                vetoMode = cms.uint32(3), # pfTICL candidate list
                vetoSrc = cms.InputTag("pfTICL")
            )
    phase2_hgcal.toModify(
      process.particleFlowBlock,
      elementImporters = _insertTrackImportersWithVeto
    )

    return process
