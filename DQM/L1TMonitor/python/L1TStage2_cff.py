import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TStage2mGMT_cfi import *

from EventFilter.L1TRawToDigi.gmtStage2Digis_cfi import *

l1tStage2online = cms.Sequence(
    l1tStage2mGMT
    )
