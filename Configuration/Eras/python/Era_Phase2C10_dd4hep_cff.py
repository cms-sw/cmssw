import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C10_cff import Phase2C10
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Phase2C10_dd4hep = cms.ModifierChain(Phase2C10, dd4hep)
