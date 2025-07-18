import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11_dd4hep_cff import Phase2C11_dd4hep
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProdPhase2

Phase2C11_dd4hep_noMkFit = cms.ModifierChain(Phase2C11_dd4hep.copyAndExclude([trackingMkFitProdPhase2]))
