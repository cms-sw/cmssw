import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Phase2C11_dd4hep = cms.ModifierChain(Phase2C11, dd4hep)
