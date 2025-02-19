import FWCore.ParameterSet.Config as cms

# L1 Emulator-Hardware comparison sequences -- Global Run
#
# J. Brooke, N. Leonardo
#
# V.M. Ghete 2010-09 - moved to standard emulator sequence ValL1Emulator_cff

# import standard emulator sequence
from L1Trigger.Configuration.ValL1Emulator_cff import *

import L1Trigger.HardwareValidation.MuonCandProducerMon_cfi
muonDtMon = L1Trigger.HardwareValidation.MuonCandProducerMon_cfi.muonCandMon.clone()
muonDtMon.DTinput = 'dttfDigis'

import L1Trigger.HardwareValidation.MuonCandProducerMon_cfi
muonCscMon = L1Trigger.HardwareValidation.MuonCandProducerMon_cfi.muonCandMon.clone()
muonCscMon.CSCinput = 'csctfDigis'


# the comparator module
from L1Trigger.HardwareValidation.L1Comparator_cfi import *
#l1compare.COMPARE_COLLS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
# ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC, LTC,GMT,GT

# subsystem sequences
deEcal = cms.Sequence(valEcalTriggerPrimitiveDigis)
deHcal = cms.Sequence(valHcalTriggerPrimitiveDigis)
deRct = cms.Sequence(valRctDigis)
deGct = cms.Sequence(valGctDigis)
deDt = cms.Sequence(valDtTriggerPrimitiveDigis)
deDttf = cms.Sequence(valCsctfTrackDigis*valDttfDigis*muonDtMon)
deCsc = cms.Sequence(valCscTriggerPrimitiveDigis)
deCsctf = cms.Sequence(valCsctfTrackDigis*valCsctfDigis*muonCscMon)
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
                                deDttf+
                                deCsc+
                                deCsctf+
                                deRpc+
                                deGmt+
                                deGt*
                                l1compare)

