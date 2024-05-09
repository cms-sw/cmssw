import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser(description='Test merge using GlobalEvFOutputModule')
parser.add_argument("--input", action="append", default=[], help="Input files")
parser.add_argument("--numThreads", help="number of threads to use", type=int, default=3)
parser.add_argument("--numFwkStreams", help="number of cmsRun streams", type=int, default=2)
parser.add_argument("--changeBranchIDLists", help="modify the branchIDLists", type=bool, default=False)
parser.add_argument("--runNumber", help="run number to use", type=int, default=100101)
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


process = cms.Process("MERGE")

if len(args.input) == 0:
    parser.error("No input files")

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(["file:"+f for f in args.input])
)

process.options = dict(numberOfThreads = args.numThreads,
                       numberOfStreams = args.numFwkStreams)

process.merge = cms.OutputModule("GlobalEvFOutputModule",
                                 SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring()),
                                 outputCommands = cms.untracked.vstring("keep *")
)

process.ep = cms.EndPath(process.merge)

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
