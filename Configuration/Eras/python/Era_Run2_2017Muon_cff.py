import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_run2_GEMSliceTest_cff import run2_GEMSliceTest

Run2_2017Muon = cms.ModifierChain(Run2_2017, run2_GEMSliceTest)
