import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
import DQMServices.Components.test.checkBooking as booking
import DQMServices.Components.test.createElements as c
import sys

process = cms.Process("TESTHARV")

folder = "TestFolder/"

process.load("DQMServices.Components.DQMFileSaver_cfi")

from DQMServices.StreamerIO.DQMProtobufReader_cff import DQMProtobufReader
process.source = DQMProtobufReader
process.source.runNumber = cms.untracked.uint32(1)
process.source.runInputDir = cms.untracked.string("./")


elements = c.createElements()

process.harvester = cms.EDAnalyzer("DummyHarvestingClient",
                                   folder = cms.untracked.string(folder),
                                   elements=cms.untracked.VPSet(*elements),
                                   cumulateRuns = cms.untracked.bool(False),
                                   cumulateLumis = cms.untracked.bool(True))

process.eff = DQMEDHarvester("DQMGenericClient",
                             efficiency = cms.vstring("eff1 \'Eff1\' Bar0 Bar1"),
                             resolution = cms.vstring(),
                             subDirs = cms.untracked.vstring(folder))

process.dqmSaver.workflow = cms.untracked.string("")
process.dqmSaver.saveByLumiSection = cms.untracked.int32(-1)
process.dqmSaver.saveByRun = cms.untracked.int32(1)
process.dqmSaver.convention = 'Online'


process.p = cms.Path(process.harvester + process.eff)
process.o = cms.EndPath(process.dqmSaver)

process.add_(cms.Service("DQMStore"))
process.DQMStore.verbose = cms.untracked.int32(5)

#process.add_(cms.Service("Tracer"))

