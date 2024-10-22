import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_run3_egamma_2023_cff import run3_egamma_2023
from Configuration.ProcessModifiers.storeZDCDigis_cff import storeZDCDigis

Run3_2023_ZDC = cms.ModifierChain(Run3, run3_egamma_2023, storeZDCDigis)
