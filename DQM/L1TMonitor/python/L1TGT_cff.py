import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TGT_cfi import *
l1tgtpath = cms.Path(cms.SequencePlaceholder("l1GtUnpack")*l1tgt)

