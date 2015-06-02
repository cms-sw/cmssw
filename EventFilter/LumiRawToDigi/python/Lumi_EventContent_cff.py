import FWCore.ParameterSet.Config as cms
#
# Author: Chris Palmer
# Date:   02.06.15
# Notes:  
#

LumiEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_lumiDigis_*_*'
    )
)
