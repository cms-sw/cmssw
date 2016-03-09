import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TStage2CaloLayer2_cfi import *
from DQM.L1TMonitor.L1TStage2mGMT_cfi import *
from DQM.L1TMonitor.L1TLayer1_cfi import *


from EventFilter.L1TRawToDigi.caloStage2Digis_cfi import *
from EventFilter.L1TRawToDigi.gmtStage2Digis_cfi import *
from EventFilter.L1TXRawToDigi.caloLayer1Stage2Digis_cfi import *

l1tStage2online = cms.Sequence(
    l1tLayer1+
    l1tStage2CaloLayer2+
    l1tStage2mGMT
    )
