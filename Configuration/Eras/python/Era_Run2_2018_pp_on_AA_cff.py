import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProd
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

Run2_2018_pp_on_AA = cms.ModifierChain(Run2_2018.copyAndExclude([trackdnn, trackingNoLoopers]), pp_on_AA, pp_on_AA_2018)

