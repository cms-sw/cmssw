import FWCore.ParameterSet.Config as cms

#from DQM.HLTEvF.HLTEventInfoClient_cfi import *

from DQMOffline.Trigger.FourVectorHLTOfflineClient_cff import *
from DQMOffline.Trigger.EgHLTOfflineClient_cfi import *
from DQMOffline.Trigger.MuonPostProcessor_cfi import *
#from DQMOffline.Trigger.BPAGPostProcessor_cff import *
from DQMOffline.Trigger.JetMETHLTOfflineClient_cfi import *
from DQMOffline.Trigger.TnPEfficiencyPostProcessor_cff import *
from DQMOffline.Trigger.HLTTauPostProcessor_cfi import *

from DQMOffline.Trigger.DQMOffline_HLT_Cert_cff import *

from DQMOffline.Trigger.topHLTDiMuonDQMClient_cfi import *


#    # use include file for dqmEnv 
#from DQMServices.Components.DQMEnvironment_cfi import *
#dqmEnv.subSystemFolder = 'HLT'



#hltOfflineDQMClient = cms.Sequence(hltFourVectorSeqClient*egHLTOffDQMClient*hLTMuonPostVal*jetMETHLTOfflineClient*tagAndProbeEfficiencyPostProcessor*HLTTauPostAnalysis*dqmOfflineHLTCert*dqmEnv)
hltOfflineDQMClient = cms.Sequence(hltFourVectorSeqClient*egHLTOffDQMClient*hLTMuonPostVal*jetMETHLTOfflineClient*tagAndProbeEfficiencyPostProcessor*HLTTauPostAnalysis*dqmOfflineHLTCert*topHLTDiMuonClient)

