import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Isotrk
# output module 
#  module alcastreamHcalIsotrkOutput = PoolOutputModule
OutALCARECOHcalCalIsoTrkFromAlCaRaw_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsoTrkFromAlCaRaw')
        ),
    outputCommands = cms.untracked.vstring( 
        'keep *_alcaHcalIsotrkFromAlCaRawProducer_HcalIsoTrack_*',
        'keep *_alcaHcalIsotrkFromAlCaRawProducer_HcalIsoTrackEvent_*',
        )
)


OutALCARECOHcalCalIsoTrkFromAlCaRaw=OutALCARECOHcalCalIsoTrkFromAlCaRaw_noDrop.clone()
OutALCARECOHcalCalIsoTrkFromAlCaRaw.outputCommands.insert(0, "drop *")
