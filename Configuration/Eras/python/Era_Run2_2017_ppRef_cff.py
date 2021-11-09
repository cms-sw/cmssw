import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProd
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

Run2_2017_ppRef = cms.ModifierChain(Run2_2017.copyAndExclude([trackingMkFitProd, trackingNoLoopers]), ppRef_2017)
