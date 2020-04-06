#! /usr/bin/env cmsRun
# Author: Izaak Neutelings (March, 2020)
#from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

# PROCESS
process = cms.Process("SiPixelVCalReader",eras.Run2_2017)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    #lastRun = cms.untracked.uint32(1),
    #timetype = cms.string('runnumber'),
    #interval = cms.uint32(1),
    firstRun = cms.untracked.uint32(313000) # select the geometry for Phase-I pixels
)

# LOAD
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.load("Configuration.StandardSequences.GeometryDB_cff")
#process.load("Configuration.StandardSequences.GeometryIdeal_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# GLOBAL TAG
from Configuration.AlCa.GlobalTag import GlobalTag
#from Configuration.AlCa.autoCond_condDBv2 import autoCond
#process.GlobalTag.globaltag = "auto:run2_data'" #autoCond['run2_design']
# In case you of conditions missing, or if you want to test a specific GT
#process.GlobalTag.globaltag = 'PRE_DES72_V6'
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
print ">>> globaltag = '%s'"%(process.GlobalTag.globaltag)

# EXTRA
outfile = "siPixelVCal_histo.root"
print ">>> outfile = '%s'"%outfile
process.TFileService = cms.Service("TFileService",
    fileName = cms.string(outfile)
)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    destinations = cms.untracked.vstring('cout')
)
process.Timing = cms.Service("Timing")

# READER
sqlfile = "sqlite_file:siPixelVCal.db"
print ">>> sqlfile = '%s'"%sqlfile
process.VCalReaderSource = cms.ESSource("PoolDBESSource",
    #BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('')
    ),
    connect = cms.string(sqlfile),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string("SiPixelVCalRcd"),
            #tag = cms.string("trivial_VCal")
            tag = cms.string("SiPixelVCal_v1")
            #tag = cms.string("SiPixelVCal_2015_v2")
        ),
        #cms.PSet(
        #    record = cms.string("SiPixelVCalSimRcd"),
        #    tag = cms.string("trivial_VCal_Sim")
        #)
    ),
)
process.myprefer = cms.ESPrefer("PoolDBESSource","VCalReaderSource")

process.VCalReader = cms.EDAnalyzer("SiPixelVCalReader",
    printDebug = cms.untracked.bool(False),
    useSimRcd = cms.bool(False)
)
#process.VCalSimReader = cms.EDAnalyzer("SiPixelVCalReader",
#    printDebug = cms.untracked.bool(False),
#    useSimRcd = cms.bool(True)                                    
#)

#process.p = cms.Path(process.VCalReader*process.VCalSimReader)
process.p = cms.Path(process.VCalReader)
