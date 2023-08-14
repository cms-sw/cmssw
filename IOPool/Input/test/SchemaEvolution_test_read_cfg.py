import FWCore.ParameterSet.Config as cms
import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ROOT Schema Evolution')

parser.add_argument("--inputFile", type=str, help="Input file name (default: SchemaEvolutionTest.root)", default="SchemaEvolutionTest.root")
parser.add_argument("--outputFileName", type=str, help="Output file name (default: SchemaEvolutionTest2.root)", default="SchemaEvolutionTest2.root")
argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

process = cms.Process("READ")

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
