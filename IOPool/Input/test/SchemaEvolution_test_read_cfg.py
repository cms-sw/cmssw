# This configuration is used to test ROOT schema evolution.

import FWCore.ParameterSet.Config as cms
import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ROOT Schema Evolution')

parser.add_argument("--inputFile", type=str, help="Input file name (default: SchemaEvolutionTest.root)", default="SchemaEvolutionTest.root")
parser.add_argument("--outputFileName", type=str, help="Output file name (default: SchemaEvolutionTest2.root)", default="SchemaEvolutionTest2.root")
parser.add_argument("--enableStreamerInfosFix", action="store_true", help="Enable service that fixes missing streamer infos")
args = parser.parse_args()

process = cms.Process("READ")

# The service named FixMissingStreamerInfos is
# tested when enabled.
#
# The version of ROOT associated with CMSSW_13_0_0
# had a bug that caused some products to be written
# to an output file with no StreamerInfo in the file.
# At some future point, if the format of one of those
# types changes then ROOT schema evolution fails.
#
# There is a workaround fix for this problem that
# involves creating a standalone ROOT file that
# contains the StreamerInfo's. Opening and closing
# that file before reading the data files will
# bring the StreamerInfos into memory and makes
# the problem files readable even if the data formats
# change.
#
# Create the "fixit" file as follows:
#
# Create a working area with release CMSSW_13_2_6
# There is nothing special about that release. We could
# have used another release for this. It was just the latest
# at the time this was done (11/1/2023). I didn't want to use
# an IB or pre-release.
# Then add the package with the relevant class definitions and dictionaries:
#     git cms-addpkg DataFormats/TestObjects
#
# Add the relevant files from the master branch:
#    git checkout official-cmssw/master DataFormats/TestObjects/interface/SchemaEvolutionTestObjects.h
#    git checkout official-cmssw/master DataFormats/TestObjects/interface/VectorVectorTop.h 
#    git checkout official-cmssw/master DataFormats/TestObjects/src/VectorVectorTop.cc
#    git checkout official-cmssw/master DataFormats/TestObjects/src/SchemaEvolutionTestObjects.cc
#    git checkout official-cmssw/master DataFormats/TestObjects/src/classes_def.xml                 
#    git checkout official-cmssw/master DataFormats/TestObjects/src/classes.h      
#
# Edit the files to use the older versions as described above and build.
#
# Note that if a release is available with the desired versions
# (class definitions and classes_def.xml), then you don't need
# to checkout the code and/or edit the code. You probably want
# to use a release at or earlier than the release you will use
# to read the file (this might not be necessary, depends on what
# has changed in ROOT...).
#
# Start root, then give the following commands at the root prompt:
#
#    root [0] auto f = TFile::Open("fixitfile.root", "NEW");
#    root [1] TClass::GetClass("edmtest::VectorVectorElementNonSplit")->GetStreamerInfo()->ForceWriteInfo(f);
#    root [2] delete f
#
# rename the output file to "fixMissingStreamerInfosUnitTest.root" and add
# to the cms-data repository for IOPool/Input.
#
# Note the test only needs the one class definition, but in real use
# cases many different types of StreamerInfos might be needed.

if args.enableStreamerInfosFix:
    process.FixMissingStreamerInfos = cms.Service("FixMissingStreamerInfos",
        fileInPath = cms.untracked.FileInPath("IOPool/Input/data/fixMissingStreamerInfosUnitTest.root")
    )

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+args.inputFile))

process.schemaEvolutionTestRead = cms.EDAnalyzer("SchemaEvolutionTestRead",
    # I stick to values exactly convertable to float
    # to avoid potential rounding issues in the test.
    expectedVectorVectorIntegralValues = cms.vint32(
        11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151
    ),
    vectorVectorTag = cms.InputTag("writeSchemaEvolutionTest", "", "PROD")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(args.outputFileName)
)

process.path = cms.Path(process.schemaEvolutionTestRead)

process.endPath = cms.EndPath(process.out)
