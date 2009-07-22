import FWCore.ParameterSet.Config as cms
#
# Name:   Scalers_EventContent_cff.py
# Author: K.Maeshima, help from Lenny Apanasevich
# Date:   03.23.2009
# Notes:  Add L1AcceptBunchCrossing and Collection (W.Badgett,2009.05.20)
#         Add Level1TriggerScalers (W.Badgett,2009.07.22)
#
# RECO content
EvtScalersRECO = cms.PSet(
   outputCommands = cms.untracked.vstring(
       'keep L1AcceptBunchCrossings_*_*_*',
       'keep L1TriggerScalerss_*_*_*',
       'keep Level1TriggerScalerss_*_*_*',
       'keep LumiScalerss_*_*_*')
)


# AOD content
EvtScalersAOD = cms.PSet(
   outputCommands = cms.untracked.vstring(
       'keep L1AcceptBunchCrossings_*_*_*',
       'keep L1TriggerScalerss_*_*_*',
       'keep Level1TriggerScalerss_*_*_*',
       'keep LumiScalerss_*_*_*')
)
