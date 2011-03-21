#!/bin/csh
cmsenv

if ("$4" == "test") then
     set connectstring = sqlite_file:test.db
else
#    set connectstring = frontier://FrontierPrep/CMS_COND_HCAL
    set connectstring = frontier://FrontierProd/CMS_COND_31X_HCAL
endif

echo "connectstring = $connectstring"

cat >! temp_dump_cfg.py <<%

import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32($3)
)

process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

process.es_pool = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string("Hcal$1Rcd"),
            tag = cms.string("$2")
        )),
      connect = cms.string('$connectstring'),
  authenticationMethod = cms.untracked.uint32(0),
)

process.dumpcond = cms.EDAnalyzer("HcalDumpConditions",
       dump = cms.untracked.vstring("$1")
)
process.p = cms.Path(process.dumpcond)
%
cmsRun temp_dump_cfg.py
rm temp_dump_cfg.py
