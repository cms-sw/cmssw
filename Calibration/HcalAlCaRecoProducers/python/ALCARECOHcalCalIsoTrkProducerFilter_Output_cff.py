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
        'keep *_alcaHcalIsotrkProducer_HcalIsoTrack_*',
        'keep *_alcaHcalIsotrkProducer_HcalIsoTrackEvent_*',
        'keep *_genParticles_*_*',
        )
)


import copy
OutALCARECOHcalCalIsoTrkProducerFilter=copy.deepcopy(OutALCARECOHcalCalIsoTrkProducerFilter_noDrop)
OutALCARECOHcalCalIsoTrkProducerFilter.outputCommands.insert(0, "drop *")
