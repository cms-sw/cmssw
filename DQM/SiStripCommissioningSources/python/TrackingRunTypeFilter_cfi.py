import FWCore.ParameterSet.Config as cms

# filter to distinguish between runs not needing or needing tracking
trackingRunTypeFilter = cms.EDFilter("SiStripCommissioningRunTypeFilter",
    runTypes = cms.vstring( 'ApvLatency', 'FineDelay' ),
    InputModuleLabel = cms.InputTag('FedChannelDigis')
)
