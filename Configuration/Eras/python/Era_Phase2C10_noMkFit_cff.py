import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C10_cff import Phase2C10
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProdPhase2

Phase2C10_noMkFit = cms.ModifierChain(Phase2C10.copyAndExclude([trackingMkFitProdPhase2]))

