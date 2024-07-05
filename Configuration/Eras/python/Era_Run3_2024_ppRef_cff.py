import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
from Configuration.Eras.Modifier_ppRef_2024_cff import ppRef_2024
from Configuration.ProcessModifiers.storeZDCDigis_cff import storeZDCDigis

Run3_2024_ppRef = cms.ModifierChain(Run3, ppRef_2017, ppRef_2024, storeZDCDigis)
