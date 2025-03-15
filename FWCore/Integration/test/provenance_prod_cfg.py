import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser(description='Write streamer output file for provenance read test')
parser.add_argument("--consumeProd2", help="add an extra producer to the job and drop on output", action="store_true")
args = parser.parse_args()


process = cms.Process("OUTPUT")

from FWCore.Modules.modules import EmptySource

runNumber = 1
eventNumber = 1
if args.consumeProd2:
    eventNumber = 2

process.source = EmptySource(firstRun = runNumber, firstEvent = eventNumber )

from FWCore.Framework.modules import AddIntsProducer, IntProducer

process.one = IntProducer(ivalue=1)
process.two = IntProducer(ivalue=2)
process.sum = AddIntsProducer(labels=['one'])
process.t = cms.Task(process.one, process.two, process.sum)

baseOutFileName = "prov"
if args.consumeProd2 :
    process.sum.untrackedLabels = ['two']
    baseOutFileName += "_extra"


from IOPool.Output.modules import PoolOutputModule

process.out = PoolOutputModule(fileName = baseOutFileName+".root",
                               outputCommands = ["drop *", "keep *_sum_*_*"])

from FWCore.Modules.modules import AsciiOutputModule
process.prnt = AsciiOutputModule(verbosity = 2, allProvenance = True)
process.e = cms.EndPath(process.out+process.prnt, process.t)
process.maxEvents.input = 1
