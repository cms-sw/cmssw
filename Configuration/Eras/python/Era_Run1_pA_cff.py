import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
#Run1 era proper is still empty
Run1_pA = cms.ModifierChain(pA_2016)

