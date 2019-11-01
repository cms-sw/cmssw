import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.TrkStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.ticlCandidateFromTrackstersProducer_cfi import ticlCandidateFromTrackstersProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer


ticlLayerTileTask = cms.Task(ticlLayerTileProducer)
ticlPFTask = cms.Task(ticlCandidateFromTrackstersProducer, pfTICLProducer)

iterTICLTask = cms.Task(ticlLayerTileTask
    ,MIPStepTask
    ,EMStepTask
    ,TrkStepTask
    ,HADStepTask
    ,ticlPFTask
    )

iterTICL = cms.Sequence(iterTICLTask)

