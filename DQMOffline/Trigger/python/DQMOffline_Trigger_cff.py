import FWCore.ParameterSet.Config as cms

# online trigger objects monitoring
from DQM.HLTEvF.HLTObjectsMonitor_cfi import *
# HLT path monitoring (per PD)
from DQMOffline.Trigger.HLTGeneralOffline_cfi import *

# lumi
from DQMOffline.Trigger.DQMOffline_LumiMontiroring_cff import *

# Egamma
from DQMOffline.Trigger.EgHLTOfflineSource_cff import *
from DQMOffline.Trigger.EgammaMonitoring_cff import * # tag-n-probe (egammaMonitorHLT + egmHLTDQMSourceExtra)
# Muon
from DQMOffline.Trigger.MuonOffline_Trigger_cff import *
# TOP
from DQMOffline.Trigger.topHLTOfflineDQM_cff import *
from DQMOffline.Trigger.TopMonitoring_cff import *
#from DQMOffline.Trigger.QuadJetAna_cfi import *
# Tau
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *
# JetMET
from DQMOffline.Trigger.JetMETHLTOfflineAnalyzer_cff import *
from DQMOffline.Trigger.JetMETPromptMonitor_cff import *

# BTV
from DQMOffline.Trigger.BTVHLTOfflineSource_cff import *
from DQMOffline.Trigger.BTaggingMonitoring_cff import *

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
from DQMOffline.Trigger.HigPhotonJetHLTOfflineSource_cfi import *  # ?!?!?!

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
# BPH
from DQMOffline.Trigger.BPHMonitor_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoHLTMon = DQMEDAnalyzer('DQMEventInfo',
    subSystemFolder = cms.untracked.string('HLT')
)
###################################################################################################
#### SEQUENCES TO BE RUN depending on the input DATAFORMAT
## on MiniAOD
## ADD here sequences/modules which rely ONLY on collections stored in the MiniAOD format
offlineHLTSourceOnMiniAOD = cms.Sequence(
)

## on AOD (w/o the need of the RECO step on-the-fly)
## ADD here sequences/modules which rely ONLY on collections stored in the AOD format
offlineHLTSourceOnAOD = cms.Sequence(
    dqmEnvHLT
    * hltResults
    * lumiMonitorHLTsequence
    * muonFullOfflineDQM
    * HLTTauDQMOffline
    * hltInclusiveVBFSource
    * higPhotonJetHLTOfflineSource # plots are filled, but I'm not sure who is really looking at them and what you can get from them ... good candidates to be moved in offlineHLTSourceOnAODextra
#    eventshapeDQMSequence * ## OBSOLETE !!!! (looks for HLT_HIQ2Top005_Centrality1030_v, HLT_HIQ2Bottom005_Centrality1030_v, etc)
#    HeavyIonUCCDQMSequence * ## OBSOLETE !!!! (looks for HLT_HIUCC100_v and HLT_HIUCC020_v)
#    hotlineDQMSequence * ## ORPHAN !!!!
    * egammaMonitorHLT 
    * exoticaMonitorHLT
    * susyMonitorHLT
    * b2gMonitorHLT
    * higgsMonitorHLT
    * smpMonitorHLT
    * topMonitorHLT
    * btagMonitorHLT 
    * bphMonitorHLT
    * hltObjectsMonitor # as online DQM, requested/suggested by TSG coordinators
    * jetmetMonitorHLT
)

## w/ the RECO step on-the-fly (to be added to offlineHLTSourceOnAOD which should run anyhow)
offlineHLTSourceWithRECO = cms.Sequence(
    hltResults
    * egHLTOffDQMSource       ## NEEDED in VALIDATION, not really in MONITORING
    * egHLTOffDQMSource_HEP17 ## NEEDED in VALIDATION, not really in MONITORING
    * jetMETHLTOfflineAnalyzer
)
####################################################################################################
# offline DQM to be run on AOD (w/o the need of the RECO step on-the-fly) only in the VALIDATION of the HLT menu based on data
# it is needed in order to have the DQM code in the release, w/o the issue of crashing the tier0
# asa the new modules in the sequence offlineHLTSourceOnAODextra are tested,
# these have to be migrated in the main offlineHLTSourceOnAOD sequence
offlineHLTSourceOnAODextra = cms.Sequence(
### POG
    btvHLTDQMSourceExtra
    * egmHLTDQMSourceExtra # empty in 10_2_0
    * jmeHLTDQMSourceExtra 
    * muoHLTDQMSourceExtra # empty in 10_2_0
    * tauHLTDQMSourceExtra # empty in 10_2_0
    * trkHLTDQMSourceExtra # empty in 10_2_0
### PAG
    * b2gHLTDQMSourceExtra
    * bphHLTDQMSourceExtra # empty in 10_2_0
    * exoHLTDQMSourceExtra
    * higHLTDQMSourceExtra
    * smpHLTDQMSourceExtra # empty in 10_2_0
    * susHLTDQMSourceExtra
    * topHLTDQMSourceExtra
    * topHLTriggerOfflineDQM # plots are filled, but I'm not sure who is really looking at them and what you can get from them ... good candidates to be moved in offlineHLTSourceOnAODextra
    * fsqHLTDQMSourceExtra # empty in 10_2_0
#    * hinHLTDQMSourceExtra
)
####################################################################################################
#### SEQUENCES TO BE RUN @Tier0 ####
### Express : not really needed
### HLTMonitor : special collections allow to monitor tracks, strip and pixel clusters, b-tagging discriminator, etc --> OfflineHLTMonitoring
### Physics PDs : monitoring vs offline collection (typically, turnON)

## DQM step on Express
offlineHLTSource4ExpressPD = cms.Sequence(
)

## DQM step on HLTMonitor
## ADD here only sequences/modules which rely on HLT collections which are stored in the HLTMonitoring stream
## and are not available in the standard RAW format
offlineHLTSource4HLTMonitorPD = cms.Sequence(
    dqmInfoHLTMon *
    lumiMonitorHLTsequence *          # lumi
    sistripMonitorHLTsequence *       # strip
    sipixelMonitorHLTsequence *       # pixel
    BTVHLTOfflineSource *             # BTV
    trackingMonitorHLT *              # tracking
    trackingMonitorHLTDisplacedJet*   # EXO : DisplacedJet Tracking 
    egmTrackingMonitorHLT *           # EGM tracking
    vertexingMonitorHLT               # vertexing
)

# sequences run @tier0 on HLTMonitor PD
OfflineHLTMonitoring = cms.Sequence(
    offlineHLTSource4HLTMonitorPD
)

# sequences run @tier0 on HLTMonitor PD w/ HI (PbPb, XeXe), pPb, ppRef
OfflineHLTMonitoringPA = cms.Sequence(
    dqmInfoHLTMon *
    trackingMonitorHLT *
    PAtrackingMonitorHLT  
)

## DQM step on physics PDs
## transient collections produced by the RECO step are allowed ;)
offlineHLTSource4physicsPD = cms.Sequence(
    offlineHLTSourceOnAOD
    * offlineHLTSourceWithRECO
)

## DQM step on special physics PDs (HI, FSQ and LowLumi, etc)
## transient collections produced by the RECO step are allowed ;)
offlineHLTSource4specialPhysicsPD = cms.Sequence(
    offlineHLTSourceOnAOD 
    * offlineHLTSourceWithRECO
    * fsqHLTOfflineSourceSequence
    * HILowLumiHLTOfflineSourceSequence
)

## DQM step on relval
offlineHLTSource4relval = cms.Sequence(
    offlineHLTSourceOnAOD
    * offlineHLTSourceWithRECO
    * offlineHLTSource4HLTMonitorPD       ## special collections (tracking, strip, pixel, etc)
    * fsqHLTOfflineSourceSequence         ## FSQ
    * HILowLumiHLTOfflineSourceSequence   ## HI
    * offlineHLTSourceOnAODextra          ## EXTRA
)
####################################################################################################
# offline DQM to be run on AOD (w/o the need of the RECO step on-the-fly) in the VALIDATION of the HLT menu based on data
# it is needed in order to have the DQM code in the release, w/o the issue of crashing the tier0
# asa the new modules in the sequence offlineHLTSourceOnAODextra are tested
# these have to be migrated in the main offlineHLTSourceOnAOD sequence
offlineValidationHLTSourceOnAOD = cms.Sequence(
    offlineHLTSourceOnAOD 
    + offlineHLTSourceOnAODextra
)
####################################################################################################


## old sequence, it should be dropped asa we are confident it is no longer used
offlineHLTSource = cms.Sequence(
    offlineHLTSource4physicsPD
)

### sequence run @tier0 (called by main DQM sequences in DQMOffline/Configuration/python/DQMOffline_cff.py) on all PDs, but HLTMonitor one
triggerOfflineDQMSource =  cms.Sequence(
    offlineHLTSource
)

# this sequence can be used by AlCa for the validation of conditions,
# because it is like offlineHLTSource (run @tier0) + offlineHLTSourceOnAODextra (meant to validate new features)
offlineValidationHLTSource = cms.Sequence(
    offlineHLTSource
    + offlineHLTSourceOnAODextra
)

