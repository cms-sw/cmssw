import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
#Run1 era proper is still empty
Run1_peripheralPbPb = cms.ModifierChain(peripheralPbPb)

