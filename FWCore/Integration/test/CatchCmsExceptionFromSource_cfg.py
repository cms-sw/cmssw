import FWCore.ParameterSet.Config as cms

import argparse
parser = argparse.ArgumentParser(description='Test catching cms::Exception from source')
parser.add_argument("--whenToThrow", type=int, default=0, help="When to throw an exception (0=do not throw, 1=constructor, 2=openFile, 3=beginJob, 4=getNextItemType, 5=readRunAuxiliary, 6=beginRun, 7=readLuminosityBlockAuxiliary, 8=beginLuminosityBlock, 9=readEvent, 10=closeFile, 11=endJob, 12=destructor)")
args = parser.parse_args()

process = cms.Process("TEST")

from FWCore.Integration.modules import ThrowingSource
process.source = ThrowingSource(whenToThrow = args.whenToThrow)

process.maxEvents.input = 1