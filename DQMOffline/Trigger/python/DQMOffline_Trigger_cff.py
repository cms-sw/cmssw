import FWCore.ParameterSet.Config as cms

# online trigger objects monitoring
from DQM.HLTEvF.HLTObjectsMonitor_cfi import *

# lumi
from DQMOffline.Trigger.DQMOffline_LumiMontiroring_cff import *
# Egamma
from DQMOffline.Trigger.HLTGeneralOffline_cfi import *
# Egamma
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *
from DQMOffline.Trigger.EgammaMonitoring_cff import * # tag-n-probe (egammaMonitorHLT + egmHLTDQMSourceExtra)
# Muon
from DQMOffline.Trigger.MuonOffline_Trigger_cff import *
# Top
#from DQMOffline.Trigger.QuadJetAna_cfi import *
# Tau
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *
# JetMET
from DQMOffline.Trigger.JetMETHLTOfflineAnalyzer_cff import *

# BTV
from DQMOffline.Trigger.BTVHLTOfflineSource_cff import *

from DQMOffline.Trigger.FSQHLTOfflineSource_cff import *
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

##hotline 
#from DQMOffline.Trigger.hotlineDQM_cfi import * # ORPHAN

##eventshape
#from DQMOffline.Trigger.eventshapeDQM_cfi import * # OBSOLETE

##UCC
#from DQMOffline.Trigger.heavyionUCCDQM_cfi import * # OBSOLETE

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

egHLTOffDQMSource_HEP17 = egHLTOffDQMSource.clone()
egHLTOffDQMSource_HEP17.subDQMDirName=cms.string('HEP17')
egHLTOffDQMSource_HEP17.doHEP =cms.bool(True)

offlineHLTSourceOnAOD = cms.Sequence(
    hltResults *
    lumiMonitorHLTsequence *
    muonFullOfflineDQM *
    HLTTauDQMOffline *
    hltInclusiveVBFSource *
    higPhotonJetHLTOfflineSource* # plots are filled, but I'm not sure who is really looking at them and what you can get from them ... good candidates to be moved in offlineHLTSourceOnAODextra
    dqmEnvHLT *
    topHLTriggerOfflineDQM * # plots are filled, but I'm not sure who is really looking at them and what you can get from them ... good candidates to be moved in offlineHLTSourceOnAODextra
#    eventshapeDQMSequence * ## OBSOLETE !!!! (looks for HLT_HIQ2Top005_Centrality1030_v, HLT_HIQ2Bottom005_Centrality1030_v, etc)
#    HeavyIonUCCDQMSequence * ## OBSOLETE !!!! (looks for HLT_HIUCC100_v and HLT_HIUCC020_v)
#    hotlineDQMSequence * ## ORPHAN !!!!
    egammaMonitorHLT * 
    exoticaMonitorHLT *
    susyMonitorHLT *
    b2gMonitorHLT *
    higgsMonitorHLT *
    smpMonitorHLT *
    topMonitorHLT *
    btagMonitorHLT *
    bphMonitorHLT *
    hltObjectsMonitor * # as online DQM, requested/suggested by TSG coordinators
    jetmetMonitorHLT
)

offlineHLTSourceWithRECO = cms.Sequence(
    jetMETHLTOfflineAnalyzer
)

# offline DQM for running in the standard RECO,DQM (in PromptReco, ReReco, relval, etc)
## THIS IS THE SEQUENCE TO BE RUN AT TIER0
## ADD here only sequences/modules which rely on transient collections produced by the RECO step
## and not stored in the AOD format
offlineHLTSource = cms.Sequence(
    offlineHLTSourceWithRECO *
    offlineHLTSourceOnAOD
)

## sequence for HI, FSQ and LowLumi
offlineHLTSourceOnAOD4LowLumi = cms.Sequence(
    offlineHLTSourceOnAOD *
    fsqHLTOfflineSourceSequence * 
    HILowLumiHLTOfflineSourceSequence
)

offlineHLTSource4LowLumi = cms.Sequence(
    offlineHLTSourceWithRECO *
    offlineHLTSourceOnAOD4LowLumi
)

# offline DQM to be run on AOD (w/o the need of the RECO step on-the-fly) only in the VALIDATION of the HLT menu based on data
# it is needed in order to have the DQM code in the release, w/o the issue of crashing the tier0
# asa the new modules in the sequence offlineHLTSourceOnAODextra are tested,
# these have to be migrated in the main offlineHLTSourceOnAOD sequence
offlineHLTSourceOnAODextra = cms.Sequence(
    egHLTOffDQMSource * ## NEEDED in VALIDATION, not really in MONITORING
    egHLTOffDQMSource_HEP17 * ## NEEDED in VALIDATION, not really in MONITORING
### POG
    btvHLTDQMSourceExtra
    * egmHLTDQMSourceExtra
    * jmeHLTDQMSourceExtra
    * muoHLTDQMSourceExtra
    * tauHLTDQMSourceExtra
    * trkHLTDQMSourceExtra
### PAG
    * b2gHLTDQMSourceExtra
    * bphHLTDQMSourceExtra
    * exoHLTDQMSourceExtra
    * higHLTDQMSourceExtra
    * smpHLTDQMSourceExtra
    * susHLTDQMSourceExtra
    * topHLTDQMSourceExtra
    * fsqHLTDQMSourceExtra
#    * hinHLTDQMSourceExtra
)

# offline DQM to be run on AOD (w/o the need of the RECO step on-the-fly) in the VALIDATION of the HLT menu based on data
# it is needed in order to have the DQM code in the release, w/o the issue of crashing the tier0
# asa the new modules in the sequence offlineHLTSourceOnAODextra are tested
# these have to be migrated in the main offlineHLTSourceOnAOD sequence
offlineValidationHLTSourceOnAOD = cms.Sequence(
    offlineHLTSourceOnAOD 
    + offlineHLTSourceOnAODextra
)

# offline DQM for the HLTMonitoring stream
## ADD here only sequences/modules which rely on HLT collections which are stored in the HLTMonitoring stream
## and are not available in the standard RAW format
dqmInfoHLTMon = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('HLT')
    )

# sequences run @tier0 on HLTMonitor PD
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

# sequences run @tier0 on HLTMonitor PD w/ HI (PbPb, XeXe), pPb, ppRef
OfflineHLTMonitoringPA = cms.Sequence(
    dqmInfoHLTMon *
    trackingMonitorHLT *
    PAtrackingMonitorHLT  
    )

### sequence run @tier0 (called by main DQM sequences in DQMOffline/Configuration/python/DQMOffline_cff.py) on all PDs, but HLTMonitor one
triggerOfflineDQMSource =  cms.Sequence(
    offlineHLTSource
)

triggerOfflineDQMSource4LowLumi =  cms.Sequence(
    offlineHLTSource4LowLumi
)
 
# this sequence can be used by AlCa for the validation of conditions,
# because it is like offlineHLTSource (run @tier0) + offlineHLTSourceOnAODextra (meant to validate new features)
offlineValidationHLTSource = cms.Sequence(
    offlineHLTSource
    + offlineHLTSourceOnAODextra
)
