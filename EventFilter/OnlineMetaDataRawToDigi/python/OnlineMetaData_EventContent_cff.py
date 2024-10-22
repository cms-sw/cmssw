import FWCore.ParameterSet.Config as cms
#
# Name:   OnlineMetaData_EventContent_cff.py
# Author: Remi Mommsen
# Date:   20.02.2018

OnlineMetaDataContent = cms.PSet(
   outputCommands = cms.untracked.vstring(
       'keep CTPPSRecord_onlineMetaDataDigis_*_*',
       'keep DCSRecord_onlineMetaDataDigis_*_*',
       'keep OnlineLuminosityRecord_onlineMetaDataDigis_*_*',
       'keep recoBeamSpot_onlineMetaDataDigis_*_*'
       )
)
