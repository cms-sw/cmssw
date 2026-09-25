import argparse

parser = argparse.ArgumentParser(description="HcalInterpolatedPulseMap sqlite database reader")

parser.add_argument("database", help="The name of the database file")
parser.add_argument("tag", help="Database tag for PoolDBESSource")
parser.add_argument("outputfile", help="The name of the output file that will contain "
                    "the binary boost archive with the HcalInterpolatedPulseMap object")
args = parser.parse_args()

print("Database file is", args.database)
print("Database tag is", '"{}"'.format(args.tag))
print("Output file is", args.outputfile)

import FWCore.ParameterSet.Config as cms

process = cms.Process('HcalInterpolatedPulseMapDBRead')

process.source = cms.Source('EmptySource') 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = "sqlite_file:{}".format(args.database)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("HcalInterpolatedPulseMapRcd"),
        tag = cms.string(args.tag)
    ))
)

process.dumper = cms.EDAnalyzer(
    'HcalInterpolatedPulseMapDBReader',
    outputFile = cms.string(args.outputfile)
)

process.p = cms.Path(process.dumper)
