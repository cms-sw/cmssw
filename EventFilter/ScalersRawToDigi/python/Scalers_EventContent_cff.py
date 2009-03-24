import FWCore.ParameterSet.Config as cms
#
# Name:   Scalers_EventContent_cff.py
# Author: K.Maeshima, help from Lenny Apanasevich
# Date:   03.23.2009
# Notes:
# 
# RAW content 
EvtScalersRAW = cms.PSet(
   outputCommands = cms.untracked.vstring(
       'keep L1TriggerScalers_*_*_*',
       'keep LumiScalers_*_*_*',
       'keep L1TriggerScalersCollection_*_*_*'
       'keep LumiScalersCollection_*_*_*)
)
# Full Event content 
EvtScalersFEVT = cms.PSet(
   outputCommands = cms.untracked.vstring(
       'keep L1TriggerScalers_*_*_*',
       'keep LumiScalers_*_*_*',
       'keep L1TriggerScalersCollection_*_*_*'
       'keep LumiScalersCollection_*_*_*)
)
# RECO content
EvtScalersRECO = cms.PSet(
   outputCommands = cms.untracked.vstring(
       'keep L1TriggerScalers_*_*_*',
       'keep LumiScalers_*_*_*',
       'keep L1TriggerScalersCollection_*_*_*'
       'keep LumiScalersCollection_*_*_*)
)
# AOD content
EvtScalersAOD = cms.PSet(
   outputCommands = cms.untracked.vstring(
       'keep L1TriggerScalers_*_*_*',
       'keep LumiScalers_*_*_*',
       'keep L1TriggerScalersCollection_*_*_*'
       'keep LumiScalersCollection_*_*_*)
)
# FEVTDEBUG content
EvtScalersFEVTDEBUG = cms.PSet(
   outputCommands = cms.untracked.vstring(
       'keep L1TriggerScalers_*_*_*',
       'keep LumiScalers_*_*_*',
       'keep L1TriggerScalersCollection_*_*_*'
       'keep LumiScalersCollection_*_*_*)
)

