import FWCore.ParameterSet.Config as cms

process = cms.Process("analyzer")

process.load("EventFilter.CSCTFRawToDigi.csctfunpacker_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/26B79166-7C00-DE11-AA81-0030486790A0.root')
)

process.csctfanalyzer = cms.EDAnalyzer("CSCTFAnalyzer",
         mbProducer     = cms.untracked.InputTag("csctfunpacker:DT"),
         lctProducer    = cms.untracked.InputTag("csctfunpacker:"),
         trackProducer  = cms.untracked.InputTag("csctfunpacker:"),
         statusProducer = cms.untracked.InputTag("csctfunpacker:")
)

process.p = cms.Path(process.csctfunpacker*process.csctfanalyzer)

