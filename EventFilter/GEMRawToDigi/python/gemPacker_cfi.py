import FWCore.ParameterSet.Config as cms
from EventFilter.GEMRawToDigi.gemPackerDefault_cfi import gemPackerDefault as _gemPackerDefault
gemPacker = _gemPackerDefault.clone()

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toModify(gemPacker, useDBEMap = True)
