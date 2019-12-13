import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_run3_ecalclustering_cff import run3_ecalclustering

Run3_ecalClustering = cms.ModifierChain(Run3,run3_ecalclustering)

