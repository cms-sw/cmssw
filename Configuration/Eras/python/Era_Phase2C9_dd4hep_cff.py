import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Phase2C9_dd4hep = cms.ModifierChain(Phase2C9, dd4hep)
