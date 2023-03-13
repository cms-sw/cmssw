import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL IsolatedBunch
# output module 
#  module alcastreamHcalIsolatedBunchOutput = PoolOutputModule

OutALCARECOHcalCalIsolatedBunchFilter_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsolatedBunchFilter')
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
OutALCARECOHcalCalIsolatedBunchFilter=copy.deepcopy(OutALCARECOHcalCalIsolatedBunchFilter_noDrop)
OutALCARECOHcalCalIsolatedBunchFilter.outputCommands.insert(0, "drop *")
