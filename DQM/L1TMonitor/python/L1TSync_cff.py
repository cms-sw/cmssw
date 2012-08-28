import FWCore.ParameterSet.Config as cms

# filter to select HLT events
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

l1tSyncHltFilter = hltHighLevel.clone(TriggerResultsTag ="TriggerResults::HLT")
l1tSyncHltFilter.throw = cms.bool(False)
l1tSyncHltFilter.HLTPaths = ['HLT_ZeroBias_v*',
                             'HLT_L1ETM30_v*',
                             'HLT_L1ETM40_v*',
                             'HLT_L1ETM70_v*',
                             'HLT_L1ETM100_v*',
                             'HLT_L1SingleEG5_v*',
                             'HLT_L1SingleEG12_v*',
                             'HLT_L1SingleJet16_v*',
                             'HLT_L1SingleJet36_v*',
                             'HLT_L1SingleMu12_v*'
                             ]



# L1 synchronization DQM module
from DQM.L1TMonitor.L1TSync_cfi import *


