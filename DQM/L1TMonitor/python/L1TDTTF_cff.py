import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TDTTF_cfi import *
l1tdttfpath = cms.Path(cms.SequencePlaceholder("l1GtUnpack")*l1tdttf)

