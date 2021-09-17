import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C12_cff import Phase2C12
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Phase2C12_dd4hep = cms.ModifierChain(Phase2C12, dd4hep)
