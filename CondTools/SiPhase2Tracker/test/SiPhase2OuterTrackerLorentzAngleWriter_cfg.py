#! /usr/bin/env cmsRun
# Author: Marco Musich (May 2020)
from __future__ import print_function
import os, shlex, shutil, getpass

import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPhase2OuterTrackerLorentzAngleWriter=dict()  
process.MessageLogger.SiPhase2OuterTrackerLorentzAngle=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
  SiPhase2OuterTrackerLorentzAngleWriter = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
  SiPhase2OuterTrackerLorentzAngle = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
)

tag = 'SiPhase2OuterTrackerLorentzAngle_T21'
suffix = 'v0'
outfile = tag+'_'+suffix+'.db'
outdb = 'sqlite_file:'+outfile

if os.path.exists(outfile):
  oldfile = outfile.replace(".db","_old.db")
  print(">>> Backing up locally existing '%s' -> '%s'"%(outfile,oldfile))
  shutil.move(outfile,oldfile)

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string(outdb)

process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D88_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21')
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('Run'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('Run'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPhase2OuterTrackerLorentzAngleRcd'),
        tag = cms.string(tag)
    ))
)

process.LAPayload = cms.EDAnalyzer("SiPhase2OuterTrackerLorentzAngleWriter")
process.LAPayload.tag = cms.string(tag)

process.p = cms.Path(process.LAPayload)

