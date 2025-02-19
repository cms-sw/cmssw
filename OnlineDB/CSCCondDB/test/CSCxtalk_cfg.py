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
        RUI01 = cms.untracked.vstring('csc_00025577_EmuRUI16_CFEB_Crosstalk_000.raw'),
        RUI00 = cms.untracked.vstring(),
        RUI03 = cms.untracked.vstring(),
        RUI02 = cms.untracked.vstring(),
        FED755 = cms.untracked.vstring(),
        FED754 = cms.untracked.vstring(),
        FED757 = cms.untracked.vstring(),
        FED756 = cms.untracked.vstring(),
        FED751 = cms.untracked.vstring(),
        RUI08 = cms.untracked.vstring(),
        FED753 = cms.untracked.vstring('RUI01'),
        FED752 = cms.untracked.vstring(),
        FED759 = cms.untracked.vstring(),
        FED758 = cms.untracked.vstring(),
        FED760 = cms.untracked.vstring(),
        firstEvent = cms.untracked.int32(0),
        #	"/localscratch2/oana/calibration_files/csc_00000685_EmuRUI01_Calib_CFEB_CrossTalk_001_070717_095027_UTC.raw"}
        dataType = cms.untracked.string('DAQ'),
        RUI09 = cms.untracked.vstring(),
        FED750 = cms.untracked.vstring()
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.analyzer = cms.EDAnalyzer("CSCCrossTalkAnalyzer",
    #change to true to send constants to DB !!
    debug = cms.untracked.bool(False),
    Verbosity = cms.untracked.int32(0)
)

process.p = cms.Path(process.muonCSCDigis*process.analyzer)
process.muonCSCDigis.UnpackStatusDigis = True
process.muonCSCDigis.isMTCCData = False
process.muonCSCDigis.UseExaminer = False

