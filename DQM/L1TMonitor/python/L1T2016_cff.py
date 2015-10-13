import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TStage2CaloLayer2_cfi import *

from EventFilter.L1TRawToDigi.caloStage2Digis_cfi import *

l1t2016Online = cms.Sequence(
            caloStage2Digis*
            l1tStage2CaloLayer2
            )
