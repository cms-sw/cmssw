import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
parser = VarParsing.VarParsing('python')
one = VarParsing.VarParsing.multiplicity.singleton
bool = VarParsing.VarParsing.varType.bool
string = VarParsing.VarParsing.varType.string
parser.register('nolegacy',             False, one, bool, "Don't run legacy harvesters")
parser.register('nomodules',             False, one, bool, "Don't run any harvesters")
parser.register('legacyoutput',         False, one, bool, "Use DQMFileSaver for output instead of DQMIO.")
parser.register('protobufinput',         False, one, bool, "Use DQMProtobufReader for input instead of DQMIO.")
parser.register('outfile',              "dqm.root", one, string, "Output file name.")
parser.parseArguments()
args = parser

process = cms.Process("HARVESTING")
process.add_(cms.Service("DQMStore"))
process.load("DQMServices.Demo.testharvester_cfi")
process.load("DQMServices.Demo.testlegacyharvester_cfi")

print args.inputFiles

if args.protobufinput:
  infile = args.inputFiles[0]
  runnr = int(infile[-6:])
  indir = "/".join(infile.split("/")[:-1])
  process.source = cms.Source("DQMStreamerReader",
    SelectEvents = cms.untracked.vstring("*"),
    runNumber = cms.untracked.uint32(runnr),
    runInputDir = cms.untracked.string(indir),
    streamLabel = cms.untracked.string("streamDQMHistograms"),
    scanOnce = cms.untracked.bool(True),
    datafnPosition = cms.untracked.uint32(4),
    minEventsPerLumi = cms.untracked.int32(1),
    delayMillis = cms.untracked.uint32(500),
    nextLumiTimeoutMillis = cms.untracked.int32(-1),
    skipFirstLumis = cms.untracked.bool(False),
    deleteDatFiles = cms.untracked.bool(False),
    endOfRunKills  = cms.untracked.bool(False),
  )

else:
  process.source = cms.Source("DQMRootSource",
                              fileNames = cms.untracked.vstring(*["file://" + f for f in args.inputFiles]))


process.harvest = cms.Sequence(process.testharvester)
process.harvestlegacy = cms.Sequence(process.testlegacyharvester)

if args.nomodules:
  pass
elif args.nolegacy:
  process.p = cms.Path(process.harvest)
else:
  process.p = cms.Path(process.harvest + process.harvestlegacy)

# legacy output
process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
  convention = cms.untracked.string('Offline'),
  fileFormat = cms.untracked.string('ROOT'),
  producer = cms.untracked.string('DQM'),
  workflow = cms.untracked.string('/Harvesting/DQMTests/DQMIO'),
  dirName = cms.untracked.string('.'),
  saveByRun = cms.untracked.int32(-1),
  saveAtJobEnd = cms.untracked.bool(True),
)

# dqmio ouptut
process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string(args.outfile))
if args.legacyoutput:
  process.e = cms.EndPath(process.dqmSaver)
else:
  process.e = cms.EndPath(process.out)

