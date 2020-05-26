import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C8_cff import Phase2C8
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Phase2C8_dd4hep = cms.ModifierChain(Phase2C8, dd4hep)
