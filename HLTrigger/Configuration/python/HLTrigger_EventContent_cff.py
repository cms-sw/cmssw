# The following comments couldn't be translated into the new config version:

# HLT trigger "bits" - hardwired EDProduct!

# HLT book-keeping product (non-debug version)

# L1 products

# beamspot

# HLT book-keeping product (debug version)

# Timing / Performance
#	,"keep edmEventTime_*_*_*"
#	,"keep HLTPerformanceInfo_*_*_*"
# DT
#	,"keep *_hltDt1DRecHits_*_*"
#	,"keep *_hltDt4DSegments_*_*"
# CSC
#	,"keep *_hltCsc2DRecHits_*_*"
#	,"keep *_hltCscSegments_*_*"
# RPC
#	,"keep *_hltRpcRecHits_*_*"
# HCAL
#	,"keep *_hltHbhereco_*_*"
#	,"keep *_hltHfreco_*_*"
#	,"keep *_hltHoreco_*_*"
# EB + EE Uncalibrated RecHits
#	,"keep *_hltEcalWeightUncalibRecHit_*_*"
# Ecal Calibrated RecHits
#	,"keep *_hltEcalPreshowerRecHit_*_*"
#	,"keep *_hltEcalRecHit_*_*"
# Clusters

# Si Pixel hits

# Si Strip hits

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

#
# Loading ALCA event content definition - needed for ALCARECO
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
        'keep *_hltGctDigis_*_*', 
        'keep *_hltL1GtObjectMap_*_*', 
        'keep *_hltL1extraParticles_*_*', 
        'keep *_hltOfflineBeamSpot_*_*')
)
#
# Corresponding three data-tier blocks
#
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
        'keep *_hltSiPixelClusters_*_*', 
        'keep *_hltSiStripClusters_*_*', 
        'keep *_hltSiPixelRecHits_*_*', 
        'keep *_hltSiStripRecHits_*_*', 
        'keep *_hltSiStripMatchedRecHits_*_*')
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
HLTriggerFEVT.outputCommands.extend(HLTDefault.outputCommands)
HLTriggerFEVT.outputCommands.extend(HLTSpecialFEVT.outputCommands)
HLTriggerRECO.outputCommands.extend(HLTDefault.outputCommands)
HLTriggerRECO.outputCommands.extend(HLTSpecialRECO.outputCommands)
HLTriggerAOD.outputCommands.extend(HLTDefault.outputCommands)
HLTriggerAOD.outputCommands.extend(HLTSpecialAOD.outputCommands)
HLTDebug.outputCommands.extend(HLTDefault.outputCommands)
HLTDebugFEVT.outputCommands.extend(HLTDebug.outputCommands)
HLTDebugFEVT.outputCommands.extend(HLTSpecialFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTDebug.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTSpecialRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTDebug.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTSpecialAOD.outputCommands)
HLTDebugFEVT.outputCommands.extend(HLTEgamma_FEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTEgamma_RECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTEgamma_AOD.outputCommands)
HLTDebugFEVT.outputCommands.extend(HLTMuonFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTMuonRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTMuonAOD.outputCommands)
HLTDebugFEVT.outputCommands.extend(HLTJetMETFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTJetMETRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTJetMETAOD.outputCommands)
HLTDebugFEVT.outputCommands.extend(HLTBTauFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTBTauRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTBTauAOD.outputCommands)
HLTDebugFEVT.outputCommands.extend(HLTXchannelFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTXchannelRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTXchannelAOD.outputCommands)
HLTDebugFEVT.outputCommands.extend(HLTSpecialFEVT.outputCommands)
HLTDebugRECO.outputCommands.extend(HLTSpecialRECO.outputCommands)
HLTDebugAOD.outputCommands.extend(HLTSpecialAOD.outputCommands)


