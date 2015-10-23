import FWCore.ParameterSet.Config as cms

# L1 Trigger DQM sequence
#
# used by DQM GUI: DQM/Integration/python/test/l1t_dqm_sourceclient-*_cfg.py
#
# standard RawToDigi sequence must be run before the L1T module, labels 
# from the standard sequence are used as default for the L1 DQM modules
# any configuration change in the unpacking must be done in l1t_dqm_sourceclient-*_cfg.py
#
# see CVS for previous authors
#
# V.M. Ghete 2011-05-25 revised version of L1 Trigger DQM
#                       

#
# DQM modules
#

# Bx Timing DQM module
from DQM.L1TMonitor.BxTiming_cfi import *

# ECAL TPG DQM module
# not run in L1T - do we need it? FIXME

# HCAL TPG DQM module 
# not run in L1T - do we need it? FIXME

# RCT DQM module 
from DQM.L1TMonitor.L1TRCT_cfi import *
l1tRctRun1 = l1tRct.clone()
l1tRct.rctSource = 'caloStage1Digis'

l1tRctfromRCT = l1tRct.clone()
l1tRctfromRCT.rctSource = 'rctDigis'
l1tRctfromRCT.HistFolder = cms.untracked.string('L1T/L1TRCT_FromRCT')

# RCT PUM DQM
from DQM.L1TMonitor.L1TPUM_cfi import *

# GCT DQM module 
from DQM.L1TMonitor.L1TGCT_cfi import *
from DQM.L1TMonitor.L1TStage1Layer2_cfi import *

# DTTPG DQM module 
# not run in L1T - do we need it? FIXME

# DTTF DQM module 
from DQM.L1TMonitor.L1TDTTF_cfi import *

# CSCTPG DQM module 
# not run in L1T - do we need it? FIXME

# CSCTF DQM module 
from DQM.L1TMonitor.L1TCSCTF_cff import *

# RPC DQM module - non-standard name of the module
from DQM.L1TMonitor.L1TRPCTF_cfi import *

# GMT DQM module 
from DQM.L1TMonitor.L1TGMT_cfi import *

# GT DQM module 
from DQM.L1TMonitor.L1TGT_cfi import *

# L1Extra DQM module
from DQM.L1TMonitor.L1ExtraDQM_cff import *

# L1 rates DQM module
from DQM.L1TMonitor.L1TRate_cfi import *

# L1 BPTX DQM module
from DQM.L1TMonitor.L1TBPTX_cfi import *

#
# other, non pure-L1 stuff
#

# scaler modules (SM and SCAL) - it uses DQM.TrigXMonitor
#
# SCAL scalers
from DQM.TrigXMonitor.L1TScalersSCAL_cfi import *
#
# SM scalers
from DQM.TrigXMonitor.L1Scalers_cfi import *
l1s.l1GtData = cms.InputTag("gtDigis")
l1s.dqmFolder = cms.untracked.string("L1T/L1Scalers_SM") 

############################################################
# Stage1 Layer1 unpacker
from EventFilter.RctRawToDigi.l1RctHwDigis_cfi import *

# Stage1 unpacker
from L1Trigger.L1TCommon.l1tRawToDigi_cfi import *
#caloStage1Digis.FedId = cms.int32(809)

# transfer stage1 format digis to legacy format digis

from L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi import *

#################################################################


#
# define sequences 
#


l1tRctSeq = cms.Sequence(
                    l1tRct + l1tRctfromRCT + l1tPUM 
                    )

l1tGctSeq = cms.Sequence(
                    l1tGct
                    )

l1tStage1Layer2Seq = cms.Sequence(
                    l1tStage1Layer2
                    )
    
# for L1ExtraDQM, one must run GGT and GMT/GT unpacker and L1Extra producer 
# with special configurations

l1ExtraDqmSeq = cms.Sequence(
                        dqmGctDigis *
                        dqmGtDigis *
                        dqmL1ExtraParticles * 
                        l1ExtraDQM
                        )

l1ExtraStage1DqmSeq = cms.Sequence(
    dqmGtDigis *
    dqmL1ExtraParticlesStage1 *
    l1ExtraDQMStage1
    )


# L1T monitor sequence 
#     modules are independent, so the order is irrelevant 

l1tMonitorOnline = cms.Sequence(
                          bxTiming +
                          l1tDttf +
                          l1tCsctf + 
                          l1tRpctf +
                          l1tGmt +
                          l1tGt + 
                          l1ExtraDqmSeq +
                          l1tBPTX +
                          l1tRate +
                          l1tRctRun1 +
                          l1tGctSeq
                          )

l1tMonitorStage1Online = cms.Sequence(
                          bxTiming +
                          l1tDttf +
                          l1tCsctf + 
                          l1tRpctf +
                          l1tGmt +
                          l1tGt +
                          rctDigis *
                          caloStage1Digis *
                          caloStage1LegacyFormatDigis*
                          l1ExtraStage1DqmSeq +
                          #l1tBPTX +
                          #l1tRate +
                          l1tStage1Layer2Seq +
                          l1tRctSeq 
                          )


# sequence for L1 Trigger DQM modules on EndPath 
# FIXME clarify why needed on EndPath

l1tMonitorEndPathSeq = cms.Sequence(
                                    l1s +
                                    l1tscalers
                                    )
                            

