import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
recoCTPPS  = cms.Sequence(totemRPLocalReconstruction)
