from builtins import range
import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester


process = cms.Process("TEST")
process.DQMStore = cms.Service("DQMStore")
process.MessageLogger = cms.Service("MessageLogger")

process.options = cms.untracked.PSet()
process.options.numberOfThreads = cms.untracked.uint32(1)
process.options.numberOfStreams = cms.untracked.uint32(1)

process.source = cms.Source("EmptySource", numberEventsInRun = cms.untracked.uint32(10),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            firstEvent = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(5))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.ana1 = DQMEDAnalyzer("DemoNormalDQMEDAnalyzer",
  folder = cms.string("DemoSubsystemNormal1"),
)

process.ana2 = DQMEDAnalyzer("DemoNormalDQMEDAnalyzer",
  folder = cms.string("DemoSubsystemNormal2"),
)

process.ana3 = DQMEDAnalyzer("DemoGlobalDQMEDAnalyzer",
  folder = cms.string("DemoSubsystemGlobal1"),
)

process.ana4 = DQMEDAnalyzer("DemoGlobalDQMEDAnalyzer",
  folder = cms.string("DemoSubsystemGlobal2"),
)

process.ana5 = DQMEDAnalyzer("DemoOneDQMEDAnalyzer",
  folder = cms.string("DemoSubsystemOne1"),
)

process.ana6 = DQMEDAnalyzer("DemoOneDQMEDAnalyzer",
  folder = cms.string("DemoSubsystemOne2"),
)

process.harv1 = DQMEDHarvester("DemoHarvester",
  target = cms.string("DemoSubsystemNormal1"),
)

process.harv2 = DQMEDHarvester("DemoHarvester",
  target = cms.string("DemoSubsystemNormal2"),
)

process.harv3 = DQMEDHarvester("DemoRunHarvester",
  target = cms.string("DemoSubsystemGlobal1"),
)

process.harv4 = DQMEDHarvester("DemoRunHarvester",
  target = cms.string("DemoSubsystemOne1"),
)

process.harv5 = DQMEDHarvester("DemoHarvester",
  target = cms.string("DemoSubsystemNormal1_lumisummary"),
  inputMEs = cms.untracked.VInputTag(
    cms.InputTag("harv1", "DQMGenerationHarvestingRun"),
    cms.InputTag("harv1", "DQMGenerationHarvestingLumi"),
  )
)

process.harv6 = DQMEDHarvester("DemoRunHarvester",
  target = cms.string("DemoSubsystemGlobal1_runsummary"),
  inputMEs = cms.untracked.VInputTag(
    cms.InputTag("harv3", "DQMGenerationHarvestingRun"),
  )
)

process.demo_reco_dqm = cms.Task(process.ana1, process.ana2, process.ana3, process.ana4, process.ana5, process.ana6)
process.demo_harvesting = cms.Task(process.harv1, process.harv2, process.harv3, process.harv4, process.harv5, process.harv6)

process.p = cms.Path(process.demo_reco_dqm, process.demo_harvesting)

process.out = cms.OutputModule(
  "DQMRootOutputModule",
  fileName = cms.untracked.string("dqm_file1.root"),
  outputCommands = cms.untracked.vstring(
    'keep *'
  )
)

process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    convention = cms.untracked.string('Offline'),
    fileFormat = cms.untracked.string('ROOT'),
    producer = cms.untracked.string('DQM'),
    workflow = cms.untracked.string('/A/B/C'),
    dirName = cms.untracked.string('.'),
)

process.o = cms.EndPath(process.out + process.dqmSaver)

#process.add_(cms.Service("Tracer",
#  dumpPathsAndConsumes = cms.untracked.bool(True)
#))

# from FWCore.ParameterSet.Utilities import convertToUnscheduled
# process = convertToUnscheduled(process)

