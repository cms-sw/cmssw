import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Isotrk
# output module 
#  module alcastreamHcalIsotrkOutput = PoolOutputModule
OutALCARECOHcalCalIsoTrkProducerFilter_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsoTrkProducerFilter')
        ),
    outputCommands = cms.untracked.vstring( 
        'keep *_HcalIsoTrack_*_*',
        'keep *_HcalIsoTrackEvent_*_*',
        )
)


import copy
OutALCARECOHcalCalIsoTrkProducerFilter=copy.deepcopy(OutALCARECOHcalCalIsoTrkProducerFilter_noDrop)
OutALCARECOHcalCalIsoTrkProducerFilter.outputCommands.insert(0, "drop *")
