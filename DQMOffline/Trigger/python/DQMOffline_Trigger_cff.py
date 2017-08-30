import FWCore.ParameterSet.Config as cms

# online trigger objects monitoring
from DQM.HLTEvF.HLTObjectsMonitor_cfi import *

# lumi
from DQMOffline.Trigger.DQMOffline_LumiMontiroring_cff import *
# Egamma
from DQMOffline.Trigger.HLTGeneralOffline_cfi import *
# Egamma
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *
from DQMOffline.Trigger.EgammaMonitoring_cff import *
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

# vertexing
from DQMOffline.Trigger.PrimaryVertexMonitoring_cff import *

# tracking
from DQMOffline.Trigger.TrackingMonitoring_cff import *
from DQMOffline.Trigger.TrackingMonitoringPA_cff import*

# hcal
from DQMOffline.Trigger.HCALMonitoring_cff import *

# strip
from DQMOffline.Trigger.SiStrip_OfflineMonitoring_cff import *

# pixel
from DQMOffline.Trigger.SiPixel_OfflineMonitoring_cff import *

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

# EXO
from DQMOffline.Trigger.ExoticaMonitoring_cff import *
# SUS
from DQMOffline.Trigger.SusyMonitoring_cff import *
# B2G
from DQMOffline.Trigger.B2GMonitoring_cff import *
# HIG
from DQMOffline.Trigger.HiggsMonitoring_cff import *
# SMP
from DQMOffline.Trigger.StandardModelMonitoring_cff import *
# TOP
from DQMOffline.Trigger.TopMonitoring_cff import *
# BTV
from DQMOffline.Trigger.BTaggingMonitoring_cff import *
# BPH
from DQMOffline.Trigger.BPHMonitor_cff import *
# remove quadJetAna
from DQMOffline.Trigger.topHLTOfflineDQM_cff import *
from DQMOffline.Trigger.JetMETPromptMonitor_cff import *

# offline DQM for running also on AOD (w/o the need of the RECO step on-the-fly)
## ADD here sequences/modules which rely ONLY on collections stored in the AOD format
offlineHLTSourceOnAOD = cms.Sequence(
    hltResults *
    lumiMonitorHLTsequence *
    egHLTOffDQMSource *
    muonFullOfflineDQM *
    HLTTauDQMOffline *
    fsqHLTOfflineSourceSequence *
    HILowLumiHLTOfflineSourceSequence *
    hltInclusiveVBFSource *
    higPhotonJetHLTOfflineSource*
    dqmEnvHLT *
    topHLTriggerOfflineDQM *
    eventshapeDQMSequence *
    HeavyIonUCCDQMSequence *
    hotlineDQMSequence *
    egammaMonitorHLT * 
    exoticaMonitorHLT *
    susyMonitorHLT *
    b2gMonitorHLT *
    higgsMonitorHLT *
    smpMonitorHLT *
    topMonitorHLT *
    btagMonitorHLT *
    bphMonitorHLT *
    hltObjectsMonitor *
    jetmetMonitorHLT
)

# offline DQM for running in the standard RECO,DQM (in PromptReco, ReReco, relval, etc)
## ADD here only sequences/modules which rely on transient collections produced by the RECO step
## and not stored in the AOD format
offlineHLTSource = cms.Sequence(
    offlineHLTSourceOnAOD
    + hcalMonitoringSequence
    + jetMETHLTOfflineAnalyzer
)

# offline DQM for the HLTMonitoring stream
## ADD here only sequences/modules which rely on HLT collections which are stored in the HLTMonitoring stream
## and are not available in the standard RAW format
dqmInfoHLTMon = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('HLT')
    )

OfflineHLTMonitoring = cms.Sequence(
    dqmInfoHLTMon *
    lumiMonitorHLTsequence * # lumi
    sistripMonitorHLTsequence * # strip
    sipixelMonitorHLTsequence * # pixel
    BTVHLTOfflineSource *
    trackingMonitorHLT * # tracking
    trackingMonitorHLTDisplacedJet* #DisplacedJet Tracking 
    egmTrackingMonitorHLT * # egm tracking
    vertexingMonitorHLT # vertexing
    )
OfflineHLTMonitoringPA = cms.Sequence(
    dqmInfoHLTMon *
    trackingMonitorHLT *
    PAtrackingMonitorHLT  
    )

triggerOfflineDQMSource =  cms.Sequence(offlineHLTSource)
 
