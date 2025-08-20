import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProdPhase2

Phase2C20I13M9_noMkFit = cms.ModifierChain(Phase2C20I13M9.copyAndExclude([trackingMkFitProdPhase2]))
