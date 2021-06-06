import FWCore.ParameterSet.Config as cms

# L1 Emulator-Hardware comparison sequences -- Global Run
#
# J. Brooke, N. Leonardo
#
# V.M. Ghete 2010-09 - moved to standard emulator sequence ValL1Emulator_cff

# import standard emulator sequence
from L1Trigger.Configuration.ValL1Emulator_cff import *

# the comparator module
from L1Trigger.HardwareValidation.L1Comparator_cfi import *

from L1Trigger.HardwareValidation.L1ComparatorforStage1_cfi import *

# subsystem sequences
deEcal = cms.Task(valEcalTriggerPrimitiveDigis)
deHcal = cms.Task(valHcalTriggerPrimitiveDigis)
deRct = cms.Task(valRctDigis)
deGct = cms.Task(valGctDigis)
deStage1Layer2 = cms.Task(
    valRctUpgradeFormatDigis
    ,valCaloStage1Digis
    #,simCaloStage1FinalDigis
    ,valCaloStage1LegacyFormatDigis
    )
deDt = cms.Task(valDtTriggerPrimitiveDigis)
deCsc = cms.Task(valCscTriggerPrimitiveDigis)
deCsctfTracks = cms.Task(valCsctfTrackDigis)
deDttf = cms.Task(valDttfDigis)
deCsctf = cms.Task(valCsctfDigis)
deRpc = cms.Task(valRpcTriggerDigis)
deGmt = cms.Task(valGmtDigis)
deGt = cms.Task(valGtDigis)
deStage1Gt = cms.Task(valStage1GtDigis)

# the sequence
L1HardwareValidationTask = cms.Task(
                                deEcal,
                                deHcal,
                                deRct,
                                deGct,
                                deDt,
                                deCsc,
                                deCsctfTracks ,
                                deDttf,
                                deCsctf,
                                deRpc,
                                deGmt,
                                deGt,
                                l1compare)
L1HardwareValidation = cms.Sequence(L1HardwareValidationTask)

L1HardwareValidationforStage1Task = cms.Task(
                                deEcal,
                                deHcal,
                                deRct,
                                deStage1Layer2,
                                deGct,
                                deDt,
                                deCsc,
                                deCsctfTracks ,
                                deDttf,
                                deCsctf,
                                deRpc,
                                deGmt,
                                deStage1Gt,
                                l1compareforstage1)
L1HardwareValidationforStage1 = cms.Sequence(L1HardwareValidationforStage1Task)

