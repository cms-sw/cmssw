import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProdPhase2

Phase2C11_noMkFit = cms.ModifierChain(Phase2C11.copyAndExclude([trackingMkFitProdPhase2]))

