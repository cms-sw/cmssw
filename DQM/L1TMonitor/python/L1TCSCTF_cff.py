import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TCSCTF_cfi import *
l1tcsctfpath = cms.Path(cms.SequencePlaceholder("l1GtUnpack")*l1tcsctf)

