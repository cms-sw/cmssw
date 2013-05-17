import FWCore.ParameterSet.Config as cms
import DQMServices.Components.test.checkBooking as booking
import DQMServices.Components.test.createElements as c

process = cms.Process("TEST")

process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')

# Input source
process.source = cms.Source("PoolSource",
                            secondaryFileNames = cms.untracked.vstring(),
                            fileNames = cms.untracked.vstring('file:dqm_file4_oldf.root'),
                            processingMode = cms.untracked.string('RunsAndLumis')
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

readRunElements1 = c.createReadRunElements()
readLumiElements1 = c.createReadLumiElements()
process.reader1 = cms.EDAnalyzer("DummyTestReadDQMStore",
                                folder = cms.untracked.string("TestFolder/"),
                                runElements = cms.untracked.VPSet(*readRunElements1),
                                lumiElements = cms.untracked.VPSet(*readLumiElements1),
                                runToCheck = cms.untracked.int32(1)
    )
readRunElements2 = c.createReadRunElements()
readLumiElements2 = c.createReadLumiElements()
process.reader2 = cms.EDAnalyzer("DummyTestReadDQMStore",
                                folder = cms.untracked.string("TestFolder/"),
                                runElements = cms.untracked.VPSet(*readRunElements2),
                                lumiElements = cms.untracked.VPSet(*readLumiElements2),
                                runToCheck = cms.untracked.int32(2)
    )


process.o = cms.EndPath(process.EDMtoME + process.reader1 + process.reader2)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

