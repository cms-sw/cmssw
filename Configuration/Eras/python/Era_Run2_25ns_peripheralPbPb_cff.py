import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb

Run2_25ns_peripheralPbPb = cms.ModifierChain(Run2_25ns, peripheralPbPb)

