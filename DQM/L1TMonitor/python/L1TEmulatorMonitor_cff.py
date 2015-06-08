import FWCore.ParameterSet.Config as cms

# L1 Emulator DQM sequence
#
# used by DQM GUI: DQM/Integration/python/test/l1temulator_dqm_sourceclient-*_cfg.py
#
# N. Leonardo 2008-02-XX initial version
#
# V.M. Ghete 2010-10-22 revised version of L1 emulator DQM
#                       proper definition of sequences

# hardware validation sequence - it runs also the L1 emulator
from L1Trigger.HardwareValidation.L1HardwareValidation_cff import *

# temporary fix for L1 GT emulator configuration in hardware validation
valGtDigis.RecordLength = cms.vint32(3, 5)
valGtDigis.AlternativeNrBxBoardDaq = 0x101
valGtDigis.AlternativeNrBxBoardEvm = 0x2
valGtDigis.AlgorithmTriggersUnprescaled = True
valGtDigis.TechnicalTriggersUnprescaled = True
valGtDigis.TechnicalTriggersVetoUnmasked = True

valStage1GtDigis.RecordLength = cms.vint32(3, 5)
valStage1GtDigis.AlternativeNrBxBoardDaq = 0x101
valStage1GtDigis.AlternativeNrBxBoardEvm = 0x2
valStage1GtDigis.AlgorithmTriggersUnprescaled = True
valStage1GtDigis.TechnicalTriggersUnprescaled = True
valStage1GtDigis.TechnicalTriggersVetoUnmasked = True


# DQM modules
from DQM.L1TMonitor.L1TDEMON_cfi import *

from DQM.L1TMonitor.L1TDEMONStage1_cfi import *

from DQM.L1TMonitor.L1TdeGCT_cfi import *

from DQM.L1TMonitor.L1TdeStage1Layer2_cfi import *

from DQM.L1TMonitor.L1TdeRCT_cfi import *
l1TdeRCT.rctSourceData = 'gctDigis'
l1TdeRCT.rctSourceEmul = 'valRctDigis'

from DQM.L1TMonitor.L1TdeCSCTF_cfi import *

from DQM.L1TMonitor.L1GtHwValidation_cff import *

# sequence for expert modules for data - emulator comparison
# the modules are independent, so uses "+"

############################################################
# Stage1 unpacker

from L1Trigger.L1TCommon.l1tRawToDigi_cfi import *

# transfer stage1 format digis to legacy format digis

from L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi import *

############################################################


l1TdeRCTSeq = cms.Sequence(
                    l1TdeRCT
                    )

l1ExpertDataVsEmulator = cms.Sequence(
                                l1TdeGCT + 
                                l1TdeCSCTF + 
                                l1GtHwValidation + 
                                l1TdeRCTSeq                 
                                )


l1EmulatorMonitor = cms.Sequence(
                            l1demon+
                            l1ExpertDataVsEmulator             
                            )

# for use in processes where hardware validation is not run
l1HwValEmulatorMonitor = cms.Sequence(
                                L1HardwareValidation*
                                l1EmulatorMonitor
                                )

# for stage1
l1ExpertDataVsEmulatorStage1 = cms.Sequence(
    caloStage1Digis*
    caloStage1LegacyFormatDigis*
    l1TdeStage1Layer2 +
    l1TdeCSCTF +
    l1Stage1GtHwValidation +
    l1TdeRCTSeq
    )

l1EmulatorMonitorStage1 = cms.Sequence(
    caloStage1Digis*
    caloStage1LegacyFormatDigis*    
    l1demonstage1+
    l1ExpertDataVsEmulatorStage1
    )

l1Stage1HwValEmulatorMonitor = cms.Sequence(
    caloStage1Digis*
    caloStage1LegacyFormatDigis*    
    L1HardwareValidationforStage1 +
    l1EmulatorMonitorStage1                            
    )
