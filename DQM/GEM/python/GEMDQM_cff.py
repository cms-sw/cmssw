import FWCore.ParameterSet.Config as cms

from DQM.GEM.GEMDQMSource_cfi import *
from DQM.GEM.GEMDQMSourceDigi_cfi import *
from DQM.GEM.GEMDQMStatusDigi_cfi import *
from DQM.GEM.GEMDQMHarvester_cfi import *

GEMDQM = cms.Sequence(
  GEMDQMSource
  *GEMDQMSourceDigi
  *GEMDQMStatusDigi
  +GEMDQMHarvester
)
