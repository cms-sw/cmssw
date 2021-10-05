import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProd

Run2_2017_ppRef = cms.ModifierChain(Run2_2017.copyAndExclude([trackingMkFitProd]), ppRef_2017)
