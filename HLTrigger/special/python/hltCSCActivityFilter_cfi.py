import FWCore.ParameterSet.Config as cms

hltCSCActivityFilter = cms.EDFilter( "HLTCSCActivityFilter",
    cscStripDigiTag = cms.InputTag("hltMuonCSCDigis","MuonCSCStripDigi"),
    inputDigis = cms.InputTag( "hltMuonCSCDigis" ),
    processDigis = cms.bool( True ),
    StationRing = cms.bool( False ),
    StationNumber = cms.int32(1),
    RingNumber = cms.int32(4),
    applyfilter = cms.untracked.bool(True)
)
