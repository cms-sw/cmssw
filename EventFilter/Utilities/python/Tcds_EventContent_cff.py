import FWCore.ParameterSet.Config as cms
#
# Author: Chris Palmer
# Date:   02.06.15
#

TcdsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_tcdsDigis_*_*'
    )
)
