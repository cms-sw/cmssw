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
