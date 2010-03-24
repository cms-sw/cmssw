import FWCore.ParameterSet.Config as cms

hltCSCAcceptBusyFilter = cms.EDFilter("HLTCSCAcceptBusyFilter",
       cscrechitsTag        = cms.InputTag( "hltCsc2DRecHits" ),
       invert               = cms.bool(True),
       maxRecHitsPerChamber = cms.uint32(200)
)
