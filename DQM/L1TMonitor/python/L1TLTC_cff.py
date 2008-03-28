import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TLTC_cfi import *
l1tltcunpack = cms.EDFilter("LTCRawToDigi")

l1tltcpath = cms.Path(l1tltcunpack*l1tltc)

