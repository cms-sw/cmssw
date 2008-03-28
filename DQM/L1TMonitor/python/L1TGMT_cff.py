import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TGMT_cfi import *
from DQM.L1TMonitor.L1TGCT_unpack_cff import *
l1tgmtpath = cms.Path(cms.SequencePlaceholder("l1GtUnpack")*l1tgmt)

