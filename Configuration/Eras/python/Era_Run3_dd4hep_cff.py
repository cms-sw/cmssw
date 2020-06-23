import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Run3_dd4hep = cms.ModifierChain(Run3, dd4hep)

