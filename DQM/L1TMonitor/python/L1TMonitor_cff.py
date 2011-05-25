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

# LTC DQM module - do these data exist, are they useful? 
# FIXME
from DQM.L1TMonitor.L1TLTC_cff import *

# ECAL TPG DQM module
# not run in L1T - do we need it? FIXME

# HCAL TPG DQM module 
# not run in L1T - do we need it? FIXME

# RCT DQM module 
from DQM.L1TMonitor.L1TRCT_cfi import *


# GCT DQM module 
from DQM.L1TMonitor.L1TGCT_cfi import *

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

# L1 synchronization DQM module
from DQM.L1TMonitor.L1TSync_cfi import *


#
# other, non pure-L1 stuff
#

# fiter used by RCT and GCT DQM modules
from HLTrigger.special.HLTTriggerTypeFilter_cfi import *
hltTriggerTypeFilter.SelectedTriggerType = 1

# scaler modules (SM and SCAL) - it uses DQM.TrigXMonitor
from DQM.L1TMonitor.L1TMonScalers_cff import *


#
# define sequences 
#

# RCT anf GCT likes to have the hltTriggerTypeFilter in the sequence 
# FIXME - talk with them about

l1tRctSeq = cms.Sequence(
                    hltTriggerTypeFilter * 
                    l1tRct
                    )

l1tGctSeq = cms.Sequence(
                    hltTriggerTypeFilter * 
                    l1tGct
                    )
# for L1ExtraDQM, one must run also the L1Extra producer

l1ExtraDqmSeq = cms.Sequence(
                        L1Extra * 
                        l1ExtraDQM
                        )

# L1T monitor sequence 
#     modules are independent, so the order is irrelevant 
#     (except l1tRctSeq and l1tGctSeq, due to the filter, which must be the last in the sequence) 

l1tMonitor = cms.Sequence(
                          bxTiming +
                          l1tLtc +
                          l1tDttf +
                          l1tCsctf + 
                          l1tRpctf +
                          l1tGmt +
                          l1tGt + 
                          l1ExtraDqmSeq +
                          l1tRate +
                          l1tSyc +
                          l1tRctSeq +
                          l1tGctSeq
                          )

# sequence for L1 Trigger DQM modules on EndPath 
# FIXME clarify why needed on EndPath

l1tMonitorEndPathSeq = cms.Sequence(
                                    l1s +
                                    l1tscalers
                                    )
                            

