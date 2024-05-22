import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser(description='Test GlobalEvFOutputModule')
parser.add_argument("--numEvents", help="number of events to process", type=int, default=10)
parser.add_argument("--startEvent", help="start event number", type=int, default=1)
parser.add_argument("--runNumber", help="run number to use", type=int, default=100101)
parser.add_argument("--numThreads", help="number of threads to use", type=int, default=3)
parser.add_argument("--numFwkStreams", help="number of cmsRun streams", type=int, default=2)
parser.add_argument("--changeBranchIDLists", help="modify the branchIDLists", type=bool, default=False)

parser.add_argument("--buBaseDir", help="BU base directory", type=str, default="ramdisk")
parser.add_argument("--fuBaseDir", help="FU base directory", type=str, default="data")
parser.add_argument("--fffBaseDir", help="FFF base directory", type=str, default=".")

args = parser.parse_args()

#try to create 'ramdisk' directory
try:
    os.makedirs(args.fffBaseDir+"/"+args.buBaseDir+"/run"+str(args.runNumber).zfill(6))
except:pass
#try to create 'data' directory
try:
  os.makedirs(args.fffBaseDir+"/"+args.fuBaseDir+"/run"+str(args.runNumber).zfill(6))
except Exception as ex:
  print(str(ex))
  pass


process = cms.Process("WRITE")

process.source = cms.Source("EmptySource",
                            firstRun=cms.untracked.uint32(args.runNumber),
                            firstEvent=cms.untracked.uint32(args.startEvent)
)

if args.numEvents != 0:
    process.maxEvents.input = args.numEvents
else:
    process.maxEvents.input = 1

process.options = dict(numberOfThreads = args.numThreads,
                       numberOfStreams = args.numFwkStreams)

process.intprod = cms.EDProducer("BranchIDListsModifierProducer",
                                 makeExtraProduct=cms.untracked.bool(args.changeBranchIDLists))

process.thing = cms.EDProducer("ThingProducer")

process.otherThing = cms.EDProducer("OtherThingProducer",
                                   thingTag=cms.InputTag("thing"))

process.t = cms.Task(
    process.intprod,
    process.thing,
    process.otherThing
)


process.filter = cms.EDFilter("PrescaleEventFilter", offset = cms.uint32(0), prescale=cms.uint32(1))
if args.numEvents == 0:
    process.filter.offset = 2
    process.filter.prescale = 4
process.p = cms.Path(process.filter)

process.streamA = cms.OutputModule("GlobalEvFOutputModule",
                                   SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("p")),
                                   outputCommands = cms.untracked.vstring("keep *")
)

process.ep = cms.EndPath(process.streamA, process.t)

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    useFileBroker = cms.untracked.bool(False),
    fileBrokerHostFromCfg = cms.untracked.bool(True),
    fileBrokerHost = cms.untracked.string("htcp40.cern.ch"),
    runNumber = cms.untracked.uint32(args.runNumber),
    baseDir = cms.untracked.string(args.fffBaseDir+"/"+args.fuBaseDir),
    buBaseDir = cms.untracked.string(args.fffBaseDir+"/"+args.buBaseDir),
    directorIsBU = cms.untracked.bool(False),
)

process.FastMonitoringService = cms.Service("FastMonitoringService",
    sleepTime = cms.untracked.int32(1)
)
