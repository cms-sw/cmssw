import FWCore.ParameterSet.Config as cms

process = cms.Process("analyzer")

process.load("EventFilter.CSCTFRawToDigi.csctfunpacker_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/data/CRAFT09/Cosmics/RAW/v1/000/112/553/C0041916-1896-DE11-948D-003048D3756A.root')
#)

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('CSCFileReader'),
    readerPset = cms.untracked.PSet(
        firstEvent  = cms.untracked.int32(0),
        tfDDUnumber = cms.untracked.int32(0),
        FED760 = cms.untracked.vstring('RUI01'),
        RUI01  = cms.untracked.vstring('/tmp/kkotov/csc_00134950_EmuRUI00_Monitor_001.raw')
#        RUI01  = cms.untracked.vstring('ev.raw')
  )
)

process.csctfanalyzer = cms.EDAnalyzer("CSCTFAnalyzer",
         mbProducer     = cms.untracked.InputTag("csctfunpacker:DT"),
         lctProducer    = cms.untracked.InputTag("csctfunpacker:"),
         trackProducer  = cms.untracked.InputTag("csctfunpacker:"),
         statusProducer = cms.untracked.InputTag("csctfunpacker:")
)

process.p = cms.Path(process.csctfunpacker*process.csctfanalyzer)

