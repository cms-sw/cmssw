import FWCore.ParameterSet.Config as cms

#from DQM.HLTEvF.HLTEventInfoClient_cfi import *

from DQMOffline.Trigger.EgHLTOfflineClient_cfi import *
from DQMOffline.Trigger.MuonPostProcessor_cff import *
#from DQMOffline.Trigger.BPAGPostProcessor_cff import *
from DQMOffline.Trigger.JetMETHLTOfflineClient_cfi import *
#from DQMOffline.Trigger.TnPEfficiencyPostProcessor_cff import *
from DQMOffline.Trigger.HLTTauPostProcessor_cfi import *
from DQMOffline.Trigger.DQMOffline_HLT_Cert_cff import *
from DQMOffline.Trigger.HLTInclusiveVBFClient_cfi import *
from DQMOffline.Trigger.FSQHLTOfflineClient_cfi import  *
from DQMOffline.Trigger.HILowLumiHLTOfflineClient_cfi import  *

from DQMOffline.Trigger.TrackingMonitoring_Client_cff import *
from DQMOffline.Trigger.TrackingMonitoringPA_Client_cff import *
from DQMOffline.Trigger.SiPixel_OfflineMonitoring_Client_cff import *

from DQMOffline.Trigger.EgammaMonitoring_Client_cff import *
from DQMOffline.Trigger.ExoticaMonitoring_Client_cff import *
from DQMOffline.Trigger.SusyMonitoring_Client_cff import *
from DQMOffline.Trigger.B2GMonitoring_Client_cff import *
from DQMOffline.Trigger.HiggsMonitoring_Client_cff import *
from DQMOffline.Trigger.StandardModelMonitoring_Client_cff import *
from DQMOffline.Trigger.TopMonitoring_Client_cff import *
from DQMOffline.Trigger.BTaggingMonitoring_Client_cff import *
from DQMOffline.Trigger.BPHMonitoring_Client_cff import *
from DQMOffline.Trigger.JetMETPromptMonitoring_Client_cff import *
from DQMOffline.Trigger.DiJetMonitor_Client_cff import *
from DQMOffline.Trigger.BTagAndProbeMonitoring_Client_cff import *

hltOfflineDQMClient = cms.Sequence(
#    hltGeneralSeqClient
    sipixelHarvesterHLTsequence
#  * trackingMonitorClientHLT
#  * trackingForElectronsMonitorClientHLT
  * trackEfficiencyMonitoringClientHLT
  * egHLTOffDQMClient
  * hltMuonPostVal
  * jetMETHLTOfflineClient
  * fsqClient
  * HiJetClient
# * tagAndProbeEfficiencyPostProcessor
  * HLTTauPostSeq
  * dqmOfflineHLTCert
  * hltInclusiveVBFClient
  * egammaClient
  * exoticaClient
  * susyClient
  * b2gClient
  * higgsClient
  * smpClient
  * topClient
  * btaggingClient
  * bphClient
  * JetMetPromClient
  * dijetClient
  * BTagAndProbeClient
)

hltOfflineDQMClientExtra = cms.Sequence(
)
