import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProdPhase2

Phase2C9_noMkFit = cms.ModifierChain(Phase2C9.copyAndExclude([trackingMkFitProdPhase2]))

