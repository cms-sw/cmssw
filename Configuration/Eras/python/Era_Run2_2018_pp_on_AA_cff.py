import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pf_badHcalMitigation_cff import pf_badHcalMitigation

Run2_2018_pp_on_AA = cms.ModifierChain(Run2_2018, pp_on_AA_2018, pf_badHcalMitigation)
