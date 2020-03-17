import FWCore.ParameterSet.Config as cms
from EventFilter.GEMRawToDigi.gemPackerDefault_cfi import gemPackerDefault as _gemPackerDefault
gemPacker = _gemPackerDefault.clone()

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM

run2_GEM_2017.toModify(gemPacker, useDBEMap = True)
run3_GEM.toModify(gemPacker, useDBEMap = True)
phase2_GEM.toModify(gemPacker, useDBEMap = False)
