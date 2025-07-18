import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2_cff import Phase2
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProdPhase2

Phase2_noMkFit = cms.ModifierChain(Phase2.copyAndExclude([trackingMkFitProdPhase2]))
