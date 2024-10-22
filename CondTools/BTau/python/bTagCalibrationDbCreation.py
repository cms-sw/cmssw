from __future__ import print_function
import os
import sys
import FWCore.ParameterSet.Config as cms

if len(sys.argv) < 2:
    raise RuntimeError('\nERROR: Need csv-filename as first argument.\n')
csv_file = sys.argv[1]
db_file = csv_file.replace('.csv', '.db')
tagger = os.path.basename(csv_file).split('.')[0]
print("Using file:", csv_file)
print("DBout into:", db_file)
print("taggername:", tagger)

process = cms.Process("BTagCalibCreator")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:' + db_file

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(1),
)

process.source = cms.Source("EmptySource")
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string(tagger),
            tag = cms.string(tagger),
            label = cms.string(tagger),
        ),
    )
)

process.dbCreator = cms.EDAnalyzer("BTagCalibrationDbCreator",
    csvFile=cms.untracked.string(csv_file),
    tagger=cms.untracked.string(tagger),
)

process.p = cms.Path(
    process.dbCreator
)


