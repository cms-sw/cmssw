import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser(description='Create files for reduced ProcessHistory test')
parser.add_argument("--version", type=str, help="CMSSW version to be used in the ProcessHistory (default is unset")
parser.add_argument("--accelerators", type=str, nargs='+', help="Propagated to process.options.accelerators (default is unset)")
parser.add_argument("--firstEvent", default=1, type=int, help="Number of first event")
parser.add_argument("--lumi", default=1, type=int, help="LuminosityBlock number")
parser.add_argument("--output", type=str, help="Output file name")

args = parser.parse_args()

class ProcessAcceleratorTest(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorTest,self).__init__()
        self._labels = ["test-one", "test-two"]
    def labels(self):
        return self._labels
    def enabledLabels(self):
        return self._labels

process = cms.Process("PROD")
if args.version:
    process._specialOverrideReleaseVersionOnlyForTesting(args.version)
if args.accelerators:
    process.add_(ProcessAcceleratorTest())
    process.options.accelerators = args.accelerators

process.maxEvents.input = 10

from FWCore.Modules.modules import EmptySource
process.source = EmptySource(
    firstEvent = args.firstEvent,
    firstLuminosityBlock = args.lumi,
)

from IOPool.Streamer.modules import EventStreamFileWriter
process.out = EventStreamFileWriter(
    fileName = args.output
)

from FWCore.Framework.modules import IntProducer
process.intProducer = IntProducer(ivalue = 42)

from FWCore.Integration.modules import ThingProducer
process.thing = ThingProducer()

process.t = cms.Task(
    process.intProducer,
    process.thing,
)
process.p = cms.Path(process.t)
process.ep = cms.EndPath(process.out)
