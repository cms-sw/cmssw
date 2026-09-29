import argparse
import FWCore.ParameterSet.Config as cms

parser = argparse.ArgumentParser()
parser.add_argument('-r','--runNumber', type=int, required=True, help='Run number')
parser.add_argument('-l','--lumiBlockNumber', type=int, required=True, help='LuminosityBlock number')
parser.add_argument('-e','--eventNumber', type=int, required=True, help='Event number')
parser.add_argument('-o','--outputFile', type=str, required=True, help='Output file name')
args = parser.parse_args()

process = cms.Process('PROD')

from FWCore.Modules.modules import EmptySource
process.source = EmptySource(
    firstRun = args.runNumber,
    firstLuminosityBlock = args.lumiBlockNumber,
    firstEvent = args.eventNumber,
    numberEventsInLuminosityBlock = 1,
    numberEventsInRun = 1
)

process.maxEvents.input = 1

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(
    fileName = args.outputFile,
    outputCommands = [
        'drop *',
        'keep *_thingWithMergeProducer_*_*'
    ]
)

from FWCore.Integration.modules import ThingWithMergeProducer
process.thingWithMergeProducer = ThingWithMergeProducer()

process.p = cms.Path(process.thingWithMergeProducer)
process.e = cms.EndPath(process.out)
