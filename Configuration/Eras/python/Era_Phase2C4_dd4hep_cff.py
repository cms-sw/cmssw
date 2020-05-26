import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C4_cff import Phase2C4
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Phase2C4_dd4hep = cms.ModifierChain(Phase2C4, dd4hep)
