import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL IsolatedBunch
# output module 
#  module alcastreamHcalIsolatedBunchOutput = PoolOutputModule

OutALCARECOHcalCalIsolatedBunchSelector_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsolatedBunchSelector')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_hbhereco_*_*',
        'keep *_ecalRecHit_*_*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep *_TriggerResults_*_*',
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep *_HBHEChannelInfo_*_*',
        'keep  FEDRawDataCollection_rawDataCollector_*_*',
        'keep  FEDRawDataCollection_source_*_*',
        )
)

import copy
OutALCARECOHcalCalIsolatedBunchSelector=copy.deepcopy(OutALCARECOHcalCalIsolatedBunchSelector_noDrop)
OutALCARECOHcalCalIsolatedBunchSelector.outputCommands.insert(0, "drop *")
