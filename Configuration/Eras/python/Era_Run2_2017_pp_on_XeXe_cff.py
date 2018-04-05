import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017

Run2_2017_pp_on_XeXe = cms.ModifierChain(Run2_2017, pp_on_XeXe_2017)
