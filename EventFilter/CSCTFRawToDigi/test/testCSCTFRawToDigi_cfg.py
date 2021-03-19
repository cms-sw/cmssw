import FWCore.ParameterSet.Config as cms  
process = cms.Process('analyzer')
process.load('EventFilter.CSCTFRawToDigi.csctfunpacker_cfi')
process.load('EventFilter.CSCTFRawToDigi.csctfpacker_cfi')
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(128) )
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')
process.source = cms.Source('EmptySource')
process.csctfsinglegen = cms.EDProducer('CSCTFSingleGen')
process.csctfpacker.lctProducer = cms.InputTag('csctfsinglegen:')
process.csctfpacker.mbProducer   = cms.InputTag('null:')
process.csctfpacker.trackProducer = cms.InputTag('null:')
process.csctfunpacker.producer = cms.InputTag('csctfpacker','CSCTFRawData')
process.csctfanalyzer = cms.EDAnalyzer('CSCTFAnalyzer',
                                       mbProducer     = cms.untracked.InputTag('csctfunpacker:DT'),
                                       lctProducer    = cms.untracked.InputTag('csctfunpacker:'),
                                       trackProducer  = cms.untracked.InputTag('csctfunpacker:'),
                                       statusProducer = cms.untracked.InputTag('csctfunpacker:')
                                   )                             
process.p = cms.Path(process.csctfsinglegen*process.csctfpacker*process.csctfunpacker*process.csctfanalyzer)
