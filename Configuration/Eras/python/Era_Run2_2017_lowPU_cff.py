import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_ppRef_cff import Run2_2017_ppRef
from Configuration.Eras.Modifier_lowPU_2017_cff import lowPU_2017

Run2_2017_lowPU = cms.ModifierChain(Run2_2017_ppRef, lowPU_2017)
