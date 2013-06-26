#!/usr/bin/env cmsRun

'''

Convert an MVA training stored in a sqlite database to a .mva binary file.

Author: Evan K. Friis (UC Davis)

'''

import FWCore.ParameterSet.Config as cms
import sys
import os

print sys.argv

db_file = sys.argv[2]
mva_file = db_file.replace('.db', '.mva')
calibration_record = os.path.splitext(os.path.basename(db_file))[0]

process = cms.Process("dump_db")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(	input = cms.untracked.int32(1) )

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
process.load = cms.ESSource(
    "PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TauTagMVAComputerRcd'),
        tag = cms.string('Train')
    )),
    connect = cms.string('sqlite:%s' % db_file),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)

process.save = cms.EDAnalyzer(
    "TauMVATrainerFileSave",
    trained = cms.untracked.bool(False),
)
setattr(process.save, calibration_record, cms.string(mva_file))

process.outpath = cms.EndPath(
    process.save
)

