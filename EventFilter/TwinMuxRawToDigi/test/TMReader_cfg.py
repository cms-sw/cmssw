import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpTwinMuxRaw")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold  = cms.untracked.string('ERROR')
                       
process.source = cms.Source("PoolSource",                 
                            fileNames = cms.untracked.vstring(  
                            '/store/data/Run2015D/SingleMuon/RAW/v1/000/258/320/00000/000EF727-DF6B-E511-9763-02163E0146D5.root',
                            )                            
                            
)

process.dttm7unpacker = cms.EDProducer("DTTM7FEDReader",
                                       #DTTM7_FED_Source = cms.InputTag("source"),
                                       DTTM7_FED_Source = cms.InputTag("rawDataCollector"),
                                       feds = cms.untracked.vint32(     1390 ),
                                       wheels = cms.untracked.vint32(   -2 ),
                                       # AMC / Sector mapping example
                                       # amcsecmap                     ( 0xF9FFFF3FFFFF ),
                                       # AmcId                         (   123456789... )
                                       # Sector                        (   -9----3----- )
                                       amcsecmap = cms.untracked.vint64( 0xFFFFFFAF9FBF ),
                                       debug = cms.untracked.bool(False),
                                       passbc0 = cms.untracked.bool(False),
)



process.RawToDigi = cms.Sequence(process.dttm7unpacker)
process.p = cms.Path(process.RawToDigi)


process.save = cms.OutputModule("PoolOutputModule",
                                fileName = cms.untracked.string('myOutputFile.root'),
                                outputCommands = cms.untracked.vstring('drop *', 
                                    'keep L1MuDTChambPhContainer_*_*_*', 
                                    'keep L1MuDTChambThContainer_*_*_*'),
)

#process.print1 = cms.OutputModule("AsciiOutputModule")
process.out = cms.EndPath(process.save)
