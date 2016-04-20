import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TdeStage2CaloLayer1_cfi import *
from DQM.L1TMonitor.L1TStage2CaloLayer2Emul_cfi import *

# These should be removed once in standard RawToDigi sequence!
from EventFilter.L1TXRawToDigi.caloLayer1Stage2Digis_cfi import *
from EventFilter.L1TRawToDigi.caloStage2Digis_cfi import *
from EventFilter.L1TRawToDigi.gmtStage2Digis_cfi import *
from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import *
from EventFilter.L1TRawToDigi.l1tRawtoDigiBMTF_cfi import *
from EventFilter.L1TRawToDigi.emtfStage2Digis_cfi import *

l1tStage2Emulator = cms.Sequence(
            l1tLayer1ValSequence
            # No valCaloStage2Digis set up yet
            # + l1tStage2CaloLayer2Emul
            )
