import FWCore.ParameterSet.Config as cms

from DQM.GEM.GEMDigiSource_cfi import *
from DQM.GEM.GEMRecHitSource_cfi import *
from DQM.GEM.GEMDAQStatusSource_cfi import *
from DQM.GEM.GEMDQMHarvester_cfi import *

GEMDQM = cms.Sequence(
  GEMDigiSource
  *GEMRecHitSource
  *GEMDAQStatusSource
  +GEMDQMHarvester
)

GEMDQMForRelval = cms.Sequence(
  GEMDigiSource
  *GEMRecHitSource
)

from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM
phase2_GEM.toModify(GEMDigiSource, digisInputLabel = "simMuonGEMDigis")
phase2_GEM.toModify(GEMDigiSource, useDBEMap = False)
phase2_GEM.toModify(GEMDAQStatusSource, useDBEMap = False)
