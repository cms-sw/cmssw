import FWCore.ParameterSet.Config as cms

# filter to select HLT events
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

l1tSyncHltFilter = hltHighLevel.clone(TriggerResultsTag ="TriggerResults::HLT")
l1tSyncHltFilter.throw = cms.bool(False)
l1tSyncHltFilter.HLTPaths = ['HLT_ZeroBias_v*',
                             'HLT_L1ETM30_v*',
                             'HLT_L1MultiJet_v*',
                             'HLT_L1SingleEG12_v',
                             'HLT_L1SingleEG5_v*',
                             'HLT_L1SingleJet16_v*',
                             'HLT_L1SingleJet36_v*',
                             'HLT_L1SingleMu10_v*',
                             'HLT_L1SingleMu20_v*'
                             ]



# L1 synchronization DQM module
from DQMOffline.L1Trigger.L1TSync_Harvest_cfi import *


