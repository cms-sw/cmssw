#!/usr/bin/env python

from __future__ import print_function
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
process = cms.Process('L1GTSUMMARY',Run2_2016)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')

##process.load('L1Trigger/L1TGlobal/debug_messages_cfi')
## process.MessageLogger.l1t_debug.l1t.limit = cms.untracked.int32(100000)
#
#process.MessageLogger.debugModules = cms.untracked.vstring('*')
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
    )

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
        '/store/data/Run2016H/ZeroBias/RAW/v1/000/283/946/00000/94A3398F-239E-E611-94A7-FA163EE85157.root'
	),
    skipEvents = cms.untracked.uint32(0)
    )

process.output =cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
	fileName = cms.untracked.string('poolout.root')
	)

process.options = cms.untracked.PSet()
## process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '90X_dataRun2_v0', '')

##### needed until prescales go into GlobalTag ########################
## from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
## process.l1conddb = cms.ESSource("PoolDBESSource",
##        CondDBSetup,
##        connect = cms.string('frontier://FrontierPrep/CMS_CONDITIONS'),
##        toGet   = cms.VPSet(
##             cms.PSet(
##                  record = cms.string('L1TGlobalPrescalesVetosRcd'),
##                  tag = cms.string("L1TGlobalPrescalesVetos_Stage2v0_hlt")
##             )
##        )
## )
## process.es_prefer_l1conddb = cms.ESPrefer( "PoolDBESSource","l1conddb")
#### done ##############################################################

process.load('L1Trigger.L1TGlobal.L1TGlobalSummary_cfi')
process.L1TGlobalSummary.AlgInputTag = cms.InputTag("gtStage2Digis")
process.L1TGlobalSummary.ExtInputTag = cms.InputTag("gtStage2Digis")
process.L1TGlobalSummary.MinBx          = cms.int32(-2)
process.L1TGlobalSummary.MaxBx          = cms.int32(2)
#process.L1TGlobalSummary.DumpRecord   = cms.bool(False) # print raw uGT record
process.L1TGlobalSummary.DumpTrigResults= cms.bool(True)
## process.l1tGlobalSummary.ReadPrescalesFromFile = cms.bool(True)
## process.l1tGlobalSummary.psFileName = cms.string("prescale_new.csv")
## process.l1tGlobalSummary.psColumn = cms.int32(0)

process.raw2digi_step = cms.Path(process.RawToDigi)
process.p = cms.Path(process.L1TGlobalSummary)

process.schedule = cms.Schedule(process.raw2digi_step,process.p)

rootout=False
if rootout:
    process.outpath = cms.EndPath(process.output)
    process.schedule.append(process.outpath)

dump=False
if dump:
    outfile = open('dump_config.py','w')
    print(process.dumpPython(), file=outfile)
    outfile.close()
