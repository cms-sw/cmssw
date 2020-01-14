import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.DQMStore = cms.Service("DQMStore", forceResetOnBeginLumi = cms.untracked.bool(True))
process.MessageLogger = cms.Service("MessageLogger")

process.load("DQMServices.Demo.test_cfi")
process.load("DQMServices.Demo.testone_cfi")
process.load("DQMServices.Demo.testonefillrun_cfi")
process.load("DQMServices.Demo.testonelumi_cfi")
process.load("DQMServices.Demo.testonelumifilllumi_cfi")
process.load("DQMServices.Demo.testglobal_cfi")
process.load("DQMServices.Demo.testlegacy_cfi")
process.load("DQMServices.Demo.testlegacyfillrun_cfi")
process.load("DQMServices.Demo.testlegacyfilllumi_cfi")
process.test_general = cms.Sequence(process.test 
                                  + process.testglobal)
process.test_one     = cms.Sequence(process.testone
                                  + process.testonefillrun)
process.test_legacy  = cms.Sequence(process.testonelumi + process.testonelumifilllumi
                                  + process.testlegacy + process.testlegacyfillrun + process.testlegacyfilllumi)

import FWCore.ParameterSet.VarParsing as VarParsing
parser = VarParsing.VarParsing('python')
one = VarParsing.VarParsing.multiplicity.singleton
int = VarParsing.VarParsing.varType.int
bool = VarParsing.VarParsing.varType.bool
string = VarParsing.VarParsing.varType.string
parser.register('nolegacy',             False, one, bool, "Don't run modules which block concurrent lumis.")
parser.register('noone',                False, one, bool, "Don't run any one modules.")
parser.register('legacyoutput',         False, one, bool, "Use DQMFileSaver for output instead of DQMIO.")
parser.register('protobufoutput',       False, one, bool, "Use DQMFileSaverPB for output instead of DQMIO.")
parser.register('onlineoutput',         False, one, bool, "Use DQMFileSaverOnline for output instead of DQMIO. This *does not* cover live mode.")
parser.register('metoedmoutput',        False, one, bool, "Use MEtoEDMConverter and PoolOutputModule for output.")
parser.register('firstLuminosityBlock', 1, one, int, "See EmptySource.")
parser.register('firstEvent',           1, one, int, "See EmptySource.")
parser.register('firstRun',             1, one, int, "See EmptySource.")
parser.register('numberEventsInRun',    100, one, int, "See EmptySource.")
parser.register('numberEventsInLuminosityBlock', 20, one, int, "See EmptySource.")
parser.register('nEvents',              100, one, int, "Total number of events.")
parser.register('nThreads',             1, one, int, "Number of threads and streams.")
parser.register('nConcurrent',          1, one, int, "Number of concurrent runs/lumis.")
parser.register('howmany',              1, one, int, "Number of MEs to book of each type.")
parser.register('outfile',              "dqm.root", one, string, "Output file name.")
parser.parseArguments()
args = parser


process.source = cms.Source("EmptySource", numberEventsInRun = cms.untracked.uint32(args.numberEventsInRun),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(args.numberEventsInLuminosityBlock),
                            firstLuminosityBlock = cms.untracked.uint32(args.firstLuminosityBlock),
                            firstEvent = cms.untracked.uint32(args.firstEvent),
                            firstRun = cms.untracked.uint32(args.firstRun))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(args.nEvents) )

process.options = cms.untracked.PSet(
  numberOfThreads = cms.untracked.uint32(args.nThreads),
  numberOfStreams = cms.untracked.uint32(args.nThreads),
  numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(args.nConcurrent),
  # Must be one for now.
  numberOfConcurrentRuns = cms.untracked.uint32(1)
)

if args.nConcurrent > 1:
  process.DQMStore.assertLegacySafe = cms.untracked.bool(False)

for mod in [process.test, process.testglobal, process.testone, process.testonefillrun, process.testonelumi, process.testonelumifilllumi, process.testlegacy, process.testlegacyfillrun, process.testlegacyfilllumi]:
  mod.howmany = args.howmany

if args.noone:
  process.p = cms.Path(process.test_general)
elif args.nolegacy:
  process.p = cms.Path(process.test_general + process.test_one)
else:
  process.p = cms.Path(process.test_general + process.test_one + process.test_legacy)

# DQMIO output
process.out = cms.OutputModule(
  "DQMRootOutputModule",
  fileName = cms.untracked.string(args.outfile),
  outputCommands = cms.untracked.vstring(
    'keep *'
  )
)

# legacy output
process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
  convention = cms.untracked.string('Offline'),
  fileFormat = cms.untracked.string('ROOT'),
  producer = cms.untracked.string('DQM'),
  workflow = cms.untracked.string('/EmptySource/DQMTests/DQMIO'),
  dirName = cms.untracked.string('.'),
  saveByRun = cms.untracked.int32(-1),
  saveAtJobEnd = cms.untracked.bool(True),
)

# protobuf output
process.pbSaver = cms.EDAnalyzer("DQMFileSaverPB",
  producer = cms.untracked.string('DQM'),
  path = cms.untracked.string('./'),
  tag = cms.untracked.string('UNKNOWN'),
  fakeFilterUnitMode = cms.untracked.bool(True),
  streamLabel = cms.untracked.string("streamDQMHistograms"),
)
# online output
process.onlineSaver = cms.EDAnalyzer("DQMFileSaverOnline",
  producer = cms.untracked.string('DQM'),
  path = cms.untracked.string('./'),
  tag = cms.untracked.string('UNKNOWN'),
  backupLumiCount = cms.untracked.int32(2),
  keepBackupLumi = cms.untracked.bool(False)
)

# MEtoEDM
process.MEtoEDMConverter = cms.EDProducer("MEtoEDMConverter",
  Name = cms.untracked.string('MEtoEDMConverter'),
  Verbosity = cms.untracked.int32(0),
  Frequency = cms.untracked.int32(50),
  MEPathToSave = cms.untracked.string('')
)
process.metoedmoutput = cms.OutputModule("PoolOutputModule",
  dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('ALCARECO'),
  ),
  fileName = cms.untracked.string(args.outfile),
  outputCommands = cms.untracked.vstring(
    'keep *'
  )
)


if args.legacyoutput:
  process.o = cms.EndPath(process.dqmSaver)
elif args.protobufoutput:
  process.o = cms.EndPath(process.pbSaver)
elif args.onlineoutput:
  process.o = cms.EndPath(process.onlineSaver)
elif args.metoedmoutput:
  process.o = cms.EndPath(process.MEtoEDMConverter + process.metoedmoutput)
else:
  process.o = cms.EndPath(process.out)


#process.Tracer = cms.Service("Tracer")
#process.DQMStore.trackME = cms.untracked.string("testlegacyfillrun")
