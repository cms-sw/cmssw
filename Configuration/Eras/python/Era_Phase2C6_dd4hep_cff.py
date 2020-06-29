import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C6_cff import Phase2C6
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Phase2C6_dd4hep = cms.ModifierChain(Phase2C6, dd4hep)
