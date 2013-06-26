import FWCore.ParameterSet.Config as cms

consumer = cms.EDFilter("ShmStreamConsumer",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep FEDRawDataCollection_*_*_*'
    ),
#    SelectEvents = cms.untracked.PSet(
#        SelectEvents = cms.vstring('p1')
#    ),
    compression_level = cms.untracked.int32(1),
    use_compression = cms.untracked.bool(True),
    max_event_size = cms.untracked.int32(25000000) ##@@ 440 FEDs @ 50kB each = 22 MB
)
