import FWCore.ParameterSet.Config as cms
import DQMServices.Components.test.checkBooking as booking
import DQMServices.Components.test.createElements as c
import sys

process = cms.Process("TEST")

process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')

# Input source
process.source = cms.Source("PoolSource",
                            secondaryFileNames = cms.untracked.vstring(),
                            fileNames = cms.untracked.vstring('file:dqm_file3_oldf.root'),
                            processingMode = cms.untracked.string('RunsAndLumis')
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

readRunElements = c.createReadRunElements()
readLumiElements = c.createReadLumiElements()

process.reader = cms.EDAnalyzer("DummyTestReadDQMStore",
                                runElements = cms.untracked.VPSet(*readRunElements),
                                lumiElements = cms.untracked.VPSet(*readLumiElements),
                                runToCheck = cms.untracked.int32(2)
    )

process.o = cms.EndPath(process.EDMtoME + process.reader)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

