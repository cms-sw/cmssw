import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("EventFilter.CSCRawToDigi.cscFrontierCablingUnpck_cff")

process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('CSCFileReader'),
    readerPset = cms.untracked.PSet(
        RUI05 = cms.untracked.vstring(),
        RUI04 = cms.untracked.vstring(),
        RUI07 = cms.untracked.vstring(),
        RUI06 = cms.untracked.vstring(),
        RUI01 = cms.untracked.vstring(),
        RUI00 = cms.untracked.vstring(),
        RUI03 = cms.untracked.vstring('/localscratch2/oana/CMSSW_2_0_0_pre6/src/OnlineDB/CSCCondDB/test/csc_00036005_EmuRUI05_Calib_CFEB_Gains_000.raw'),
        RUI02 = cms.untracked.vstring(),
        FED755 = cms.untracked.vstring(),
        FED754 = cms.untracked.vstring(),
        FED757 = cms.untracked.vstring(),
        FED756 = cms.untracked.vstring(),
        FED751 = cms.untracked.vstring(),
        RUI08 = cms.untracked.vstring(),
        FED753 = cms.untracked.vstring('RUI03'),
        FED752 = cms.untracked.vstring(),
        FED759 = cms.untracked.vstring(),
        FED758 = cms.untracked.vstring(),
        FED760 = cms.untracked.vstring(),
        firstEvent = cms.untracked.int32(0),
        RUI16 = cms.untracked.vstring(),
        RUI14 = cms.untracked.vstring(),
        RUI15 = cms.untracked.vstring(),
        #                                                      // "csc_00000684_EmuRUI01_Calib_CFEB_Gains_001_070717_094214_UTC.raw",
        #						       //"csc_00000684_EmuRUI01_Calib_CFEB_Gains_002_070717_094214_UTC.raw"}
        dataType = cms.untracked.string('DAQ'),
        RUI09 = cms.untracked.vstring(),
        FED750 = cms.untracked.vstring()
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.analyzer = cms.EDAnalyzer("CSCGainAnalyzer",
    #change to true to send constants to DB !!
    debug = cms.untracked.bool(False),
    Verbosity = cms.untracked.int32(0)
)

process.p = cms.Path(process.muonCSCDigis*process.analyzer)
process.muonCSCDigis.UnpackStatusDigis = True
process.muonCSCDigis.isMTCCData = False
process.muonCSCDigis.ExaminerMask = 0x164BF3F6
process.muonCSCDigis.UseExaminer = True

