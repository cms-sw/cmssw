import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11I13_cff import Phase2C11I13
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProdPhase2

Phase2C11I13_noMkFit = cms.ModifierChain(Phase2C11I13.copyAndExclude([trackingMkFitProdPhase2]))
