# The following comments couldn't be translated into the new config version:

# HLT trigger "bits" - hardwired EDProduct!

# HLT book-keeping product (non-debug version)

# L1 products

# beamspot

# HLT book-keeping product (debug version)

# Timing / Performance

# L1

# DT

# CSC

# RPC

#HCAL

#EB + EE Uncalibrated RecHits

#Ecal Calibrated RecHits

#Clusters

#Si Pixel hits

#Si Strip hits

#save digis sim link and trigger infos

#
# DEBUG must include the non-debug stuff
#

#

#

#

#

#

#
# Special collections
#include "HLTrigger/special/data/HLTSpecial_EventContent.cff"
#

import FWCore.ParameterSet.Config as cms

# Adding ALCA event content needed for ALCARECO
from HLTrigger.special.HLTSpecial_EventContent_cff import *
#
# Egamma collections
from HLTrigger.Egamma.HLTEgamma_EventContent_cff import *
#
# Muon collections
from HLTrigger.Muon.HLTMuon_EventContent_cff import *
#
# JetMET collections
from HLTrigger.JetMET.HLTJetMET_EventContent_cff import *
#
# BTau collections
from HLTrigger.btau.HLTBTau_EventContent_cff import *
#
# Xchannel collections
from HLTrigger.xchannel.HLTXchannel_EventContent_cff import *
#
# EventContent for HLT related products.
#
# This cff file exports the following six EventContent blocks:
#   HLTriggerFEVT  HLTriggerRECO  HLTriggerAOD (without DEBUG products)
#   HLTDebugFEVT   HLTDebugRECO   HLTDebugAOD  (with    DEBUG products)
# All else is internal and should not be used directly by non-HLT users.
#
#
# Default (non-debug) products
#
HLTDefault = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*', 
        'keep *_hltGtDigis_*_*', 
        'keep *_hltL1GtObjectMap_*_*', 
        'keep *_hltL1extraParticles_*_*', 
        'keep *_hltOfflineBeamSpot_*_*')
)
#
# Corresponding three data-tier blocks
#
# Full Event content
HLTriggerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_hlt*_*_*')
)
# RECO content
HLTriggerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_hlt*_*_*')
)
# AOD content
HLTriggerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_hlt*_*_*')
)
#
# DEBUG: Additional products for Trigger debugging
#
HLTDebug = cms.PSet(
    outputCommands = cms.untracked.vstring('keep triggerTriggerEventWithRefs_*_*_*', 
        'keep edmEventTime_*_*_*', 
        'keep HLTPerformanceInfo_*_*_*', 
        'keep *_hltGctDigis_*_*', 
        'keep *_hltDt1DRecHits_*_*', 
        'keep *_hltDt4DSegments_*_*', 
        'keep *_hltCsc2DRecHits_*_*', 
        'keep *_hltCscSegments_*_*', 
        'keep *_hltRpcRecHits_*_*', 
        'keep *_hltHbhereco_*_*', 
        'keep *_hltHfreco_*_*', 
        'keep *_hltHoreco_*_*', 
        'keep *_hltEcalWeightUncalibRecHit_*_*', 
        'keep *_hltEcalPreshowerRecHit_*_*', 
        'keep *_hltEcalRecHit_*_*', 
        'keep *_hltSiPixelClusters_*_*', 
        'keep *_hltSiStripClusters_*_*', 
        'keep *_hltSiPixelRecHits_*_*', 
        'keep *_hltSiStripRecHits_*_*', 
        'keep *_hltSiStripMatchedRecHits_*_*', 
        'keep StripDigiSimLinkedmDetSetVector_hltMuonCSCDigis_*_*', 
        'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_hltMuonCSCDigis_*_*')
)
#
# Corresponding three data-tier blocks with DEBUG
#
# Full Event content
HLTDebugFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_hlt*_*_*')
)
# RECO content
HLTDebugRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_hlt*_*_*')
)
# AOD content
HLTDebugAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_hlt*_*_*')
)
#
# Add what trigger groups want:
# -> products (now for FEVT/RECO/AOD DEBUG case only!)
# -> TriggerSummaryAOD configuration
TriggerSummaryAOD = cms.PSet(
    collections = cms.VInputTag(),
    filters = cms.VInputTag()
)
HLTriggerFEVT.outputCommands.extend(HLTDefault.outputCommands)
HLTriggerFEVT.outputCommands.extend(HLTSpecialFEVT.outputCommands)
HLTriggerRECO.outputCommands.extend(HLTDefault.outputCommands)
HLTriggerRECO.outputCommands.extend(HLTSpecialRECO.outputCommands)
HLTriggerAOD.outputCommands.extend(HLTDefault.outputCommands)
HLTriggerAOD.outputCommands.extend(HLTSpecialAOD.outputCommands)
HLTDebug.outputCommands.extend(HLTDefault.outputCommands)
HLTDebugFEVT.outputCommands.extend(HLTDebug.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTDebug.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTDebug.outputCommands)
HLTDebugFEVT.outputCommands.extend(HLTEgamma_FEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTEgamma_RECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTEgamma_AOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTEgamma_AOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTEgamma_AOD.triggerFilters)
HLTDebugFEVT.outputCommands.extend(HLTMuonFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTMuonRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTMuonAOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTMuonAOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTMuonAOD.triggerFilters)
HLTDebugFEVT.outputCommands.extend(HLTJetMETFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTJetMETRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTJetMETAOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTJetMETAOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTJetMETAOD.triggerFilters)
HLTDebugFEVT.outputCommands.extend(HLTBTauFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTBTauRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTBTauAOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTBTauAOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTBTauAOD.triggerFilters)
HLTDebugFEVT.outputCommands.extend(HLTXchannelFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTXchannelRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTXchannelAOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTXchannelAOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTXchannelAOD.triggerFilters)
HLTDebugFEVT.outputCommands.extend(HLTSpecialFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTSpecialRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTSpecialAOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTSpecialAOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTSpecialAOD.triggerFilters)

