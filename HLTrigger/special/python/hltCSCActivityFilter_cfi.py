import FWCore.ParameterSet.Config as cms

hltCSCActivityFilter = cms.EDFilter( "HLTCSCActivityFilter",
    cscStripDigiTag = cms.InputTag("hltMuonCSCDigis","MuonCSCStripDigi"),
    skipStationRing = cms.bool( True ),
    skipStationNumber = cms.int32(1),
    skipRingNumber = cms.int32(4),
    saveTags = cms.bool( False )
)
