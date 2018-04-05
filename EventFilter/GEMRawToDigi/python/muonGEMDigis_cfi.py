import FWCore.ParameterSet.Config as cms
from EventFilter.GEMRawToDigi.muonGEMDigisDefault_cfi import muonGEMDigisDefault as _muonGEMDigisDefault
muonGEMDigis = _muonGEMDigisDefault.clone()

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toModify(muonGEMDigis, useDBEMap = True)
