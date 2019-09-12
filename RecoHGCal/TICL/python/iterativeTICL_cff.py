import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import *
from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.TrkStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

iterTICLTask = cms.Task(ticlLayerTileTask
    ,MIPStepTask
    ,TrkStepTask
    ,EMStepTask
    ,HADStepTask
    )

iterTICL = cms.Sequence(iterTICLTask)

