import FWCore.ParameterSet.Config as cms

print "\n!!!!! EventFilter/L1TRawToDigi/python/l1tRawtoDigiBMTF_cfi.py will be depricated soon.  Please migrate to using the unpacker EventFilter/L1TRawToDigi/python/bmtfDigis_cfi. !!!!!\n"


BMTFStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage2::BMTFSetup"),
    InputLabel = cms.InputTag("rawDataCollector"),
    FedIds = cms.vint32(1376,1377),
    FWId = cms.uint32(1),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8),
    lenAMCHeader = cms.untracked.int32(8),
    lenAMCTrailer = cms.untracked.int32(0),
    lenAMC13Header = cms.untracked.int32(8),
    lenAMC13Trailer = cms.untracked.int32(8)
)
