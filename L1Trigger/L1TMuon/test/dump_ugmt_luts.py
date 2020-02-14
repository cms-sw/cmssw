# This can be used to generate uGMT LUTs from parameters in the fakeGmtParams_cff.py file
# or dump LUTs from the conditions DB when given a record+tag+(optional)snapshotTime combination
# The LUTs are dumped as .txt files in the lut_dump directory, which needs to be created before running.

import FWCore.ParameterSet.Config as cms

process = cms.Process("L1MicroGMTEmulator")

process.load("FWCore.MessageService.MessageLogger_cfi")


process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

## use this to produce LUTs from the config parameters in fakeGmtParams_cff
process.load('L1Trigger.L1TMuon.fakeGmtParams_cff')

## use this to extract LUTs from the CondDB record + tag + snapshotTime(optional)
#from CondCore.CondDB.CondDB_cfi import CondDB
#CondDB.connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
#process.l1ugmtdb = cms.ESSource("PoolDBESSource",
#    CondDB,
#    toGet   = cms.VPSet(
#        cms.PSet(
#            record = cms.string('L1TMuonGlobalParamsRcd'),
#            tag = cms.string("L1TMuonGlobalParams_Stage2v0_hlt"),
#            #tag = cms.string("L1TMuonGlobalParams_static_v91.12"),
#            #tag = cms.string("L1TMuonGlobalParams_static_v94.6.1"),
#            #tag = cms.string("L1TMuonGlobalParams_Stage2v0_2018_mc"),
#            #snapshotTime = cms.string("2017-09-20 23:59:59.000")
#        )
#    )
#)

process.dumper = cms.EDAnalyzer("L1TMicroGMTLUTDumper",
    out_directory = cms.string("lut_dump"),
)

process.dumpPath = cms.Path( process.dumper )
process.schedule = cms.Schedule(process.dumpPath)
