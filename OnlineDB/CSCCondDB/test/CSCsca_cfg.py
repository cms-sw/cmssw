import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('CSCFileReader'),
    readerPset = cms.untracked.PSet(
        RUI05 = cms.untracked.vstring(),
        RUI04 = cms.untracked.vstring(),
        RUI07 = cms.untracked.vstring(),
        RUI06 = cms.untracked.vstring(),
        RUI01 = cms.untracked.vstring(),
        RUI00 = cms.untracked.vstring('/localscratch2/oana/calibration_files/csc_00014348_EmuRUI00_Calib_CFEB_SCAPed_000.raw'),
        RUI03 = cms.untracked.vstring(),
        RUI02 = cms.untracked.vstring(),
        FED755 = cms.untracked.vstring(),
        FED754 = cms.untracked.vstring(),
        FED757 = cms.untracked.vstring(),
        FED756 = cms.untracked.vstring(),
        FED751 = cms.untracked.vstring(),
        RUI08 = cms.untracked.vstring(),
        FED753 = cms.untracked.vstring(),
        FED752 = cms.untracked.vstring(),
        FED759 = cms.untracked.vstring(),
        FED758 = cms.untracked.vstring(),
        FED760 = cms.untracked.vstring(),
        firstEvent = cms.untracked.int32(0),
        dataType = cms.untracked.string('DAQ'),
        RUI09 = cms.untracked.vstring(),
        FED750 = cms.untracked.vstring('RUI00')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.cscunpacker = cms.EDProducer("CSCDCCUnpacker",
    #untracked bool PrintEventNumber = false
    Debug = cms.untracked.bool(False),
    Verbosity = cms.untracked.int32(0),
    InputObjects = cms.InputTag("source"),
    theMappingFile = cms.FileInPath('OnlineDB/CSCCondDB/test/csc_slice_test_map.txt')
)

process.analyzer = cms.EDAnalyzer("CSCscaAnalyzer",
    #change to true to send constants to DB !!
    debug = cms.untracked.bool(False),
    Verbosity = cms.untracked.int32(0)
)

process.p = cms.Path(process.cscunpacker*process.analyzer)

# foo bar baz
# oC1UZRFaJN3ys
