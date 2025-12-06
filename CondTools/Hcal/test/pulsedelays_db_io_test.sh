#!/bin/bash -ex

inputfile=$(edmFileInPath CondTools/Hcal/data/hcalpulsedelays.txt)
cat << \EOF > temp_pulsedelays_to_db.py

import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("TODB",eras.Run3)
process.load("CondCore.CondDB.CondDB_cfi")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond["phase1_2022_realistic"] 

process.load('Configuration.StandardSequences.Services_cff')
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = "sqlite_file:HcalPulseDelays_V00_test.db"

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.es_ascii = cms.ESSource("HcalTextCalibrations",
   input = cms.VPSet(
       cms.PSet(
           object = cms.string("PulseDelays"),
           file = cms.FileInPath("CondTools/Hcal/data/hcalpulsedelays.txt")
      )
   )
)
process.es_prefer = cms.ESPrefer('HcalTextCalibrations','es_ascii')

process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string("HcalPulseDelaysRcd"),
        tag = cms.string("HcalPulseDelays_test_tag")
    ))
)

process.mytest = cms.EDAnalyzer("HcalPulseDelaysPopConAnalyzer",
    record = cms.string('HcalPulseDelaysRcd'),
    loggingOn = cms.untracked.bool(True),
    SinceAppendMode = cms.bool(True),
    Source = cms.PSet(
        IOVRun = cms.untracked.uint32(1)
    )
)
process.p = cms.Path(process.mytest)
EOF

cmsRun temp_pulsedelays_to_db.py
rm temp_pulsedelays_to_db.py

cat << \EOF > temp_pulsedelays_from_db.py

import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("FROMDB",eras.Run3)
process.load("CondCore.CondDB.CondDB_cfi")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond["phase1_2022_realistic"] 

process.load('Configuration.StandardSequences.Services_cff')
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = "sqlite_file:HcalPulseDelays_V00_test.db"

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("HcalPulseDelaysRcd"),
        tag = cms.string("HcalPulseDelays_test_tag")
    ))
)

process.es_prefer = cms.ESPrefer("PoolDBESSource","")

process.dumpcond = cms.EDAnalyzer("HcalDumpConditions",
    dump = cms.untracked.vstring("PulseDelays")
)
process.p = cms.Path(process.dumpcond)
EOF

cmsRun temp_pulsedelays_from_db.py
rm temp_pulsedelays_from_db.py

# Ingnore raw channel ids in the comparison
rm -f db_delay_reference.txt db_delay_dump.txt
awk '{print $1, $2, $3, $4, $5, $6}' $inputfile > db_delay_reference.txt
awk '{print $1, $2, $3, $4, $5, $6}' DumpPulseDelays_Run1.txt > db_delay_dump.txt
diff db_delay_dump.txt db_delay_reference.txt
