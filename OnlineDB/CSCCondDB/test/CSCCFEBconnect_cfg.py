import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")

process.load("EventFilter.CSCRawToDigi.cscFrontierCablingUnpck_cff")

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('CSCFileReader'),
    readerPset = cms.untracked.PSet(
        RUI05 = cms.untracked.vstring(),
        RUI04 = cms.untracked.vstring(),
        RUI07 = cms.untracked.vstring(),
        RUI06 = cms.untracked.vstring(),
        RUI01 = cms.untracked.vstring(),
        RUI00 = cms.untracked.vstring(),
        RUI03 = cms.untracked.vstring('csc_00000000_EmuRUI13_Calib_CFEB_CrossTalk_000.raw'),
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
        dataType = cms.untracked.string('DAQ'),
        RUI09 = cms.untracked.vstring(),
        FED750 = cms.untracked.vstring()
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.analyzer = cms.EDAnalyzer("CSCCFEBConnectivityAnalyzer",
    #change to true to send constants to DB !!
    debug = cms.untracked.bool(False),
    Verbosity = cms.untracked.int32(0)
)

process.p = cms.Path(process.muonCSCDigis*process.analyzer)
process.muonCSCDigis.UnpackStatusDigis = True
process.muonCSCDigis.isMTCCData = False
process.muonCSCDigis.UseExaminer = False

