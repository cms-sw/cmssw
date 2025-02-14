import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C12_cff import Phase2C12
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProdPhase2

Phase2C12_noMkFit = cms.ModifierChain(Phase2C12.copyAndExclude([trackingMkFitProdPhase2]))

