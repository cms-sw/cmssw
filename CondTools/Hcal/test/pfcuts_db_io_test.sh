#!/bin/bash -ex

inputfile=$(edmFileInPath CondTools/Hcal/data/hcalpfcuts.txt)
cat << \EOF > temp_pfcuts_to_db.py

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
process.CondDB.connect = "sqlite_file:HcalPFCuts_V00_test.db"

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
           object = cms.string("PFCuts"),
           file = cms.FileInPath("CondTools/Hcal/data/hcalpfcuts.txt")
      )
   )
)
process.es_prefer = cms.ESPrefer('HcalTextCalibrations','es_ascii')

process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string("HcalPFCutsRcd"),
        tag = cms.string("HcalPFCuts_test_tag")
    ))
)

process.mytest = cms.EDAnalyzer("HcalPFCutsPopConAnalyzer",
    record = cms.string('HcalPFCutsRcd'),
    loggingOn = cms.untracked.bool(True),
    SinceAppendMode = cms.bool(True),
    Source = cms.PSet(
        IOVRun = cms.untracked.uint32(1)
    )
)
process.p = cms.Path(process.mytest)
EOF

cmsRun temp_pfcuts_to_db.py
rm temp_pfcuts_to_db.py

cat << \EOF > temp_pfcuts_from_db.py

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
process.CondDB.connect = "sqlite_file:HcalPFCuts_V00_test.db"

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
        record = cms.string("HcalPFCutsRcd"),
        tag = cms.string("HcalPFCuts_test_tag")
    ))
)

process.es_prefer = cms.ESPrefer("PoolDBESSource","")

process.dumpcond = cms.EDAnalyzer("HcalDumpConditions",
    dump = cms.untracked.vstring("PFCuts")
)
process.p = cms.Path(process.dumpcond)
EOF

cmsRun temp_pfcuts_from_db.py
rm temp_pfcuts_from_db.py

diff DumpPFCuts_Run1.txt $inputfile
