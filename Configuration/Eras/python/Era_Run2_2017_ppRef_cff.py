import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017

Run2_2017_ppRef = cms.ModifierChain(Run2_2017, ppRef_2017)
