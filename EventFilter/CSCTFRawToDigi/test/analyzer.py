import FWCore.ParameterSet.Config as cms

process = cms.Process("analyzer")

process.load("EventFilter.CSCTFRawToDigi.csctfunpacker_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/CRAFT09/Cosmics/RAW/v1/000/112/553/1CF42D9B-1496-DE11-9A39-000423D99996.root',
                                      '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/553/AACD18ED-1A96-DE11-AB83-000423D9880C.root',
                                      '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/553/C0041916-1896-DE11-948D-003048D3756A.root')
)

process.csctfanalyzer = cms.EDAnalyzer("CSCTFAnalyzer",
         mbProducer     = cms.untracked.InputTag("csctfunpacker:DT"),
         lctProducer    = cms.untracked.InputTag("csctfunpacker:"),
         trackProducer  = cms.untracked.InputTag("csctfunpacker:"),
         statusProducer = cms.untracked.InputTag("csctfunpacker:")
)

process.p = cms.Path(process.csctfunpacker*process.csctfanalyzer)

