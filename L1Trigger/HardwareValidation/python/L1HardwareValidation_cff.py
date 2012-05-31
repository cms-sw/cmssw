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

# subsystem sequences
deEcal = cms.Sequence(valEcalTriggerPrimitiveDigis)
deHcal = cms.Sequence(valHcalTriggerPrimitiveDigis)
deRct = cms.Sequence(valRctDigis)
deGct = cms.Sequence(valGctDigis)
deDt = cms.Sequence(valDtTriggerPrimitiveDigis)
deCsc = cms.Sequence(valCscTriggerPrimitiveDigis)
deCsctfTracks = cms.Sequence(valCsctfTrackDigis)
deDttf = cms.Sequence(valDttfDigis)
deCsctf = cms.Sequence(valCsctfDigis)
deRpc = cms.Sequence(valRpcTriggerDigis)
deGmt = cms.Sequence(valGmtDigis)
deGt = cms.Sequence(valGtDigis)

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

