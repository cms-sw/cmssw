import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TLayer1_cfi import *
from DQM.L1TMonitor.L1TStage2CaloLayer2_cfi import *
from DQM.L1TMonitor.L1TStage2uGMT_cfi import *
from DQM.L1TMonitor.L1TStage2uGT_cfi import *
from DQM.L1TMonitor.L1TStage2BMTF_cfi import *
from DQM.L1TMonitor.L1TStage2EMTF_cfi import *

from EventFilter.L1TXRawToDigi.caloLayer1Stage2Digis_cfi import *
from EventFilter.L1TRawToDigi.caloStage2Digis_cfi import *
from EventFilter.L1TRawToDigi.gmtStage2Digis_cfi import *
from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import *
from EventFilter.L1TRawToDigi.l1tRawtoDigiBMTF_cfi import *
from EventFilter.L1TRawToDigi.emtfStage2Digis_cfi import *

l1tStage2online = cms.Sequence(
    l1tLayer1 + 
    l1tStage2CaloLayer2 +
    l1tStage2uGMT +
    l1tStage2uGt +
    l1tStage2Bmtf +
    l1tStage2Emtf
)

