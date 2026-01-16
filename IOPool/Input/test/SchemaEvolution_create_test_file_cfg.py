import FWCore.ParameterSet.Config as cms

import argparse

parser = argparse.ArgumentParser(description='Create schema evolution test file')
parser.add_argument("--splitLevel", type=int, default=99, help="Split level for PoolOutputModule")

args = parser.parse_args()

process = cms.Process("PROD")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.writeSchemaEvolutionTest = cms.EDProducer("SchemaEvolutionTestWrite",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values. Note only values exactly convertible to
    # float are used to avoid precision and rounding issues in
    # in comparisons.
    testIntegralValues = cms.vint32(
        11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(f'SchemaEvolutionTest_splitLevel{args.splitLevel}.root'),
    splitLevel = cms.untracked.int32(args.splitLevel)
)

process.path = cms.Path(process.writeSchemaEvolutionTest)
process.endPath = cms.EndPath(process.out)
