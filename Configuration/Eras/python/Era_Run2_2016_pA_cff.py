import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

Run2_2016_pA = cms.ModifierChain(Run2_2016.copyAndExclude([trackingNoLoopers]), pA_2016)

