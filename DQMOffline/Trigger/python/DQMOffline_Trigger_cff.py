import FWCore.ParameterSet.Config as cms

# Egamma
from DQMOffline.Trigger.HLTGeneralOffline_cfi import *

from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *
# Muon
from DQMOffline.Trigger.MuonOffline_Trigger_cff import *
# Top
#from DQMOffline.Trigger.QuadJetAna_cfi import *
# Tau
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *
# JetMET
from DQMOffline.Trigger.JetMETHLTOfflineAnalyzer_cff import *

# BTV
from DQMOffline.Trigger.BTVHLTOfflineSource_cfi import *

from DQMOffline.Trigger.FSQHLTOfflineSource_cfi import *
from DQMOffline.Trigger.HILowLumiHLTOfflineSource_cfi import *

# TnP
#from DQMOffline.Trigger.TnPEfficiency_cff import *
# Inclusive VBF
from DQMOffline.Trigger.HLTInclusiveVBFSource_cfi import *

# tracking
from DQMOffline.Trigger.TrackingMonitoring_cff import *

# strip
from DQMOffline.Trigger.SiStrip_OfflineMonitoring_cff import *

# photon jet
from DQMOffline.Trigger.HigPhotonJetHLTOfflineSource_cfi import * 

#hotline 
from DQMOffline.Trigger.hotlineDQM_cfi import *

#eventshape
from DQMOffline.Trigger.eventshapeDQM_cfi import *

#UCC
from DQMOffline.Trigger.heavyionUCCDQM_cfi import *

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvHLT= DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvHLT.subSystemFolder = 'HLT'


# remove quadJetAna
from DQMOffline.Trigger.topHLTOfflineDQM_cff import *
offlineHLTSource = cms.Sequence(
    hltResults *
    egHLTOffDQMSource *
    muonFullOfflineDQM *
    HLTTauDQMOffline *
    jetMETHLTOfflineAnalyzer *
    fsqHLTOfflineSourceSequence *
    HILowLumiHLTOfflineSourceSequence *
    hltInclusiveVBFSource *
    higPhotonJetHLTOfflineSource*
    dqmEnvHLT *
    topHLTriggerOfflineDQM *
    eventshapeDQMSequence *
    HeavyIonUCCDQMSequence *
    hotlineDQMSequence
    )

# offline DQM for the HLTMonitoring stream
dqmInfoHLTMon = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('HLT')
    )

OfflineHLTMonitoring = cms.Sequence(
    dqmInfoHLTMon *
    sistripMonitorHLTsequence * 
    BTVHLTOfflineSource *
    trackingMonitorHLT
    )

triggerOfflineDQMSource =  cms.Sequence(offlineHLTSource)
 
