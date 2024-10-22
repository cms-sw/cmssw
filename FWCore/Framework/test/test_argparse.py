import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser(description='Test argparse')
parser.add_argument("--maxEvents", help="max events to process", type=int, default=1)
# same as cmsRun arguments; but ignored, just print values
parser.add_argument("-j", "--jobreport", help="file name for job report file", type=str, default="UNSET")
parser.add_argument("-e", "--enablejobreport", help="enable job report file(s)", default="UNSET", action="store_true")
parser.add_argument("-m", "--mode", help="job mode for MessageLogger", type=str, default="UNSET")
parser.add_argument("-n", "--numThreads", help="number of threads", type=str, default="UNSET")
parser.add_argument("-s", "--sizeOfStackForThreadsInKB", help="size of thread stack in KB", type=str, default="UNSET")
parser.add_argument("--strict", help="strict parsing", default="UNSET", action="store_true")
parser.add_argument("-c", "--command", help="config passed in as string", type=str, default="UNSET")
args = parser.parse_args()

print(args)

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")

process.maxEvents.input = args.maxEvents

print('TestArgParse')
