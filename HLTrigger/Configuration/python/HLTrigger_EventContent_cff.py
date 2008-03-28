# The following comments couldn't be translated into the new config version:

# HLT trigger "bits" - hardwired EDProduct!

# HLT book-keeping products

# Timing / Performance

# HLT trigger "bits" - hardwired EDProduct!

# HLT book-keeping products

# Timing / Performance

# HLT trigger "bits" - hardwired EDProduct!

# HLT book-keeping products

#

#

#

#

#

#

import FWCore.ParameterSet.Config as cms

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
# Special collections
from HLTrigger.special.HLTSpecial_EventContent_cff import *
#
# EventContent for HLT related products.
# See below for HLT-related physics collections,
# which are specific to groups of triggers.
#
# Full Event content
HLTriggerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*')
)
# RECO content
HLTriggerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*')
)
# AOD content
HLTriggerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*')
)
#  AOD only
TriggerSummaryAOD = cms.PSet(
    collections = cms.VInputTag(),
    filters = cms.VInputTag()
)
HLTriggerFEVT.outputCommands.extend(HLTEgamma_FEVT.outputCommands)
HLTriggerRECO.outputCommands.extend(HLTEgamma_RECO.outputCommands)
HLTriggerAOD.outputCommands.extend(HLTEgamma_AOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTEgamma_AOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTEgamma_AOD.triggerFilters)
HLTriggerFEVT.outputCommands.extend(HLTMuonFEVT.outputCommands)
HLTriggerRECO.outputCommands.extend(HLTMuonRECO.outputCommands)
HLTriggerAOD.outputCommands.extend(HLTMuonAOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTMuonAOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTMuonAOD.triggerFilters)
HLTriggerFEVT.outputCommands.extend(HLTJetMETFEVT.outputCommands)
HLTriggerRECO.outputCommands.extend(HLTJetMETRECO.outputCommands)
HLTriggerAOD.outputCommands.extend(HLTJetMETAOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTJetMETAOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTJetMETAOD.triggerFilters)
HLTriggerFEVT.outputCommands.extend(HLTBTauFEVT.outputCommands)
HLTriggerRECO.outputCommands.extend(HLTBTauRECO.outputCommands)
HLTriggerAOD.outputCommands.extend(HLTBTauAOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTBTauAOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTBTauAOD.triggerFilters)
HLTriggerFEVT.outputCommands.extend(HLTXchannelFEVT.outputCommands)
HLTriggerRECO.outputCommands.extend(HLTXchannelRECO.outputCommands)
HLTriggerAOD.outputCommands.extend(HLTXchannelAOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTXchannelAOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTXchannelAOD.triggerFilters)
HLTriggerFEVT.outputCommands.extend(HLTSpecialFEVT.outputCommands)
HLTriggerRECO.outputCommands.extend(HLTSpecialRECO.outputCommands)
HLTriggerAOD.outputCommands.extend(HLTSpecialAOD.outputCommands)
TriggerSummaryAOD.collections.extend(HLTSpecialAOD.triggerCollections)
TriggerSummaryAOD.filters.extend(HLTSpecialAOD.triggerFilters)

