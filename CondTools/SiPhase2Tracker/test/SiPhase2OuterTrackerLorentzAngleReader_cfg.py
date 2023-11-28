#! /usr/bin/env cmsRun
# Author: Marco Musich (October 2021)
import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPhase2OuterTrackerLorentzAngleReader=dict()  
process.MessageLogger.SiPhase2OuterTrackerLorentzAngle=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
  SiPhase2OuterTrackerLorentzAngleReader = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
  SiPhase2OuterTrackerLorentzAngle = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
)

###################################################################
# A data source must always be defined.
# We don't need it, so here's a dummy one.
###################################################################
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

###################################################################
# Input data
###################################################################
tag = 'SiPhase2OuterTrackerLorentzAngle_T21'
suffix = 'v0'
inFile = tag+'_'+suffix+'.db'
inDB = 'sqlite_file:'+inFile

process.load("CondCore.CondDB.CondDB_cfi")
# input database (in this case the local sqlite file)
process.CondDB.connect = inDB

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      process.CondDB,
                                      DumpStat=cms.untracked.bool(True),
                                      toGet = cms.VPSet(cms.PSet(record = cms.string("SiPhase2OuterTrackerLorentzAngleRcd"),
                                                                 tag = cms.string(tag))
                                                       )
                                      )

###################################################################
# check the ES data getter
###################################################################
process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string(' SiPhase2OuterTrackerLorentzAngleRcd'),
        data = cms.vstring('SiPhase2OuterTrackerLorentzAngle')
    )),
    verbose = cms.untracked.bool(True)
)

###################################################################
# Payload reader
###################################################################
import CondTools.SiPhase2Tracker.siPhase2OuterTrackerLorentzAngleReader_cfi as _mod
process.LAPayloadReader = _mod.siPhase2OuterTrackerLorentzAngleReader.clone(printDebug = 10,
                                                                            label = "")

###################################################################
# Path
###################################################################
process.p = cms.Path(process.LAPayloadReader)

