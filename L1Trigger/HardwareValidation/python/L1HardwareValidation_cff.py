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
deEcal = cms.Sequence(valEcalTriggerPrimitiveDigis)
deHcal = cms.Sequence(valHcalTriggerPrimitiveDigis)
deRct = cms.Sequence(valRctDigis)
deGct = cms.Sequence(valGctDigis)
deStage1Layer2 = cms.Sequence(
    simRctUpgradeFormatDigis
    *simCaloStage1Digis
    #*simCaloStage1FinalDigis
    *valCaloStage1LegacyFormatDigis
    )
deDt = cms.Sequence(valDtTriggerPrimitiveDigis)
deCsc = cms.Sequence(valCscTriggerPrimitiveDigis)
deCsctfTracks = cms.Sequence(valCsctfTrackDigis)
deDttf = cms.Sequence(valDttfDigis)
deCsctf = cms.Sequence(valCsctfDigis)
deRpc = cms.Sequence(valRpcTriggerDigis)
deGmt = cms.Sequence(valGmtDigis)
deGt = cms.Sequence(valGtDigis)
deStage1Gt = cms.Sequence(valStage1GtDigis)

# the sequence
L1HardwareValidation = cms.Sequence(
                                deEcal+
                                deHcal+
                                deRct+
                                deGct+
                                deDt+
                                deCsc+
                                deCsctfTracks +
                                deDttf+
                                deCsctf+
                                deRpc+
                                deGmt+
                                deGt*
                                l1compare)

L1HardwareValidationforStage1 = cms.Sequence(
                                deEcal+
                                deHcal+
                                deRct+
                                deStage1Layer2+
                                deDt+
                                deCsc+
                                deCsctfTracks +
                                deDttf+
                                deCsctf+
                                deRpc+
                                deGmt+
                                deStage1Gt*
                                l1compareforstage1)



