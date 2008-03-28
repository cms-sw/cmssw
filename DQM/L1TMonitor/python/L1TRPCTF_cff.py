import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TRPCTF_cfi import *
l1trpctfpath = cms.Path(cms.SequencePlaceholder("l1GtUnpack")*l1trpctf)

