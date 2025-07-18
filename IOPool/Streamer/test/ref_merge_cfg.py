import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ConditionalTasks.')

parser.add_argument("--inFile1", help="first file to read")
parser.add_argument("--inFile2", help="second file to read")
parser.add_argument("--outFile", help="name of output file", default="ref_merge.root")

args = parser.parse_args()

process = cms.Process("MERGE")

from IOPool.Streamer.modules import NewEventStreamFileReader
process.source = NewEventStreamFileReader(fileNames = [f"file:{args.inFile1}",
                                         f"file:{args.inFile2}"]
                            )

from IOPool.Streamer.modules import EventStreamFileWriter
process.out = EventStreamFileWriter(fileName = args.outFile)

from FWCore.Integration.modules import OtherThingAnalyzer
process.tester = OtherThingAnalyzer(other = ("d","testUserTag"))

process.o = cms.EndPath(process.out+process.tester)
