import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("DaqSource",
    maxEvents = cms.untracked.int32(-1),
    pset = cms.PSet(
        fileNames = cms.untracked.vstring('runs/wire3.bin')
    ),
    reader = cms.string('CSCFileReader')
)

process.cscunpacker = cms.EDProducer("CSCDCCUnpacker",
    #untracked bool PrintEventNumber = false
    Debug = cms.untracked.bool(False),
    theMappingFile = cms.FileInPath('OnlineDB/CSCCondDB/test/csc_slice_test_map.txt')
)

process.analyzer = cms.EDAnalyzer("CSCAFEBdacAnalyzer")

process.p = cms.Path(process.cscunpacker*process.analyzer)

