import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer

from RecoHGCal.TICL.CLUE3DEM_cff import *
from RecoHGCal.TICL.CLUE3DHAD_cff import *
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer
from RecoHGCal.TICL.tracksterSelectionTf_cfi import *

from RecoHGCal.TICL.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoHGCal.TICL.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer
from RecoHGCal.Configuration.RecoHGCal_EventContent_cff import customiseForTICLv5EventContent
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels, ticlIterLabelsMerge


def customiseForTICLv5(process):

    process.ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

    process.ticlIterationsTask = cms.Task(
        ticlCLUE3DEMStepTask,
        ticlCLUE3DHADStepTask,
    )

    process.ticlTracksterLinks = _tracksterLinksProducer.clone()
    process.ticlTracksterLinksTask = cms.Task(process.ticlTracksterLinks)

    process.ticlCandidate = _ticlCandidateProducer.clone()
    process.ticlCandidateTask = cms.Task(process.ticlCandidate)

    
    process.iterTICLTask = cms.Task(process.ticlLayerTileTask,
                                     process.ticlIterationsTask,
                                     process.ticlTracksterLinksTask,
                                     process.ticlCandidateTask)
    process.mergeTICLTask = cms.Task()

    process = customiseForTICLv5EventContent(process)

    return process
