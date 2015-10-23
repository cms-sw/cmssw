import FWCore.ParameterSet.Config as cms
#
# Name:   Scalers_EventContent_cff.py
# Author: K.Maeshima, help from Lenny Apanasevich
# Date:   03.23.2009
# Notes:  Add L1AcceptBunchCrossing and Collection (W.Badgett,2009.05.20)
#         Add Level1TriggerScalers (W.Badgett,2009.07.22)
#         Add BeamSpots and DcsStatuss (W.Badgett, 2009.10.23)
#
# RECO content
EvtScalersRECO = cms.PSet(
   outputCommands = cms.untracked.vstring(
       #'keep L1AcceptBunchCrossings_*_*_*',
       'keep L1AcceptBunchCrossings_scalersRawToDigi_*_*',
       'keep L1TriggerScalerss_scalersRawToDigi_*_*',
       'keep Level1TriggerScalerss_scalersRawToDigi_*_*',
       'keep LumiScalerss_scalersRawToDigi_*_*',
       'keep BeamSpotOnlines_scalersRawToDigi_*_*',
       'keep DcsStatuss_scalersRawToDigi_*_*',
       'keep DcsStatuss_hltScalersRawToDigi_*_*'
       )
)


# AOD content
EvtScalersAOD = cms.PSet(
   outputCommands = cms.untracked.vstring(
       #'keep L1AcceptBunchCrossings_*_*_*',
       'keep L1AcceptBunchCrossings_scalersRawToDigi_*_*',
       'keep L1TriggerScalerss_scalersRawToDigi_*_*',
       'keep Level1TriggerScalerss_scalersRawToDigi_*_*',
       'keep LumiScalerss_scalersRawToDigi_*_*',
       'keep BeamSpotOnlines_scalersRawToDigi_*_*',
       'keep DcsStatuss_scalersRawToDigi_*_*'
       )
)
