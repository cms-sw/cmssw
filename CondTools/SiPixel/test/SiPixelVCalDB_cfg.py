#! /usr/bin/env cmsRun
# Author: Izaak Neutelings (March, 2020)
#from __future__ import print_function
#import os
import os, shlex, shutil, getpass
#import subprocess
import FWCore.ParameterSet.Config as cms

# LOAD PROCESS
process = cms.Process("SiPixelInclusiveBuilder")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = cms.untracked.vstring("cout")
process.MessageLogger.cout = cms.untracked.PSet(threshold = cms.untracked.string("INFO"))
#process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.GeometryIdeal_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("Configuration.StandardSequences.GeometryDB_cff")
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")
#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")
#process.load("CondCore.DBCommon.CondDBCommon_cfi") # deprecated
process.load("CondCore.CondDB.CondDB_cfi")

# GLOBAL TAG
from Configuration.AlCa.GlobalTag import GlobalTag
#from Configuration.AlCa.autoCond_condDBv2 import autoCond
#process.GlobalTag.globaltag = "auto:run2_data'" #autoCond['run2_design']
# In case you of conditions missing, or if you want to test a specific GT
#process.GlobalTag.globaltag = 'PRE_DES72_V6'
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
print ">>> globaltag = '%s'"%(process.GlobalTag.globaltag)

# BASIC SETTING
process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# RUN AS USER
user = getpass.getuser()
#try:
#    user = os.environ["USER"]
#except KeyError:
#    user = subprocess.call('whoami')
#    # user = commands.getoutput('whoami') 
#file = "/tmp/" + user + "/SiPixelVCal.db"
file = "siPixelVCal.db"
sqlfile = "sqlite_file:" + file
print ">>> Uploading as user %s into file %s, i.e. %s"%(user,file,sqlfile)

# BACK UP DATABASE FILE
if os.path.exists("siPixelVCal.db"):
  oldfile = file.replace(".db","_old.db")
  print ">>> Backing up locally existing '%s' -> '%s'"%(file,oldfile)
  shutil.move(file,oldfile)

# DATABASE CONNNECTION AND INPUT TAGS
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('.'),
        connectionRetrialPeriod = cms.untracked.int32(10),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(1),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(0),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False)
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string(sqlfile),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('SiPixelVCalRcd'),
            tag = cms.string('SiPixelVCal_v1')
        ),
        #cms.PSet(
        #    record = cms.string('SiPixelVCalSimRcd'),
        #    tag = cms.string('SiPixelVCalSim_v1')
        #),
    )
)

# VCAL TO NUMBER OF ELECTRONS OBJECT
# https://github.com/cms-analysis/DPGAnalysis-SiPixelTools/blob/1232a8c0ef3abe7b78c757887138089706e0499a/GainCalibration/test/vcal-irradiation-factors.txt
slope     = 47.
slope_L1  = 50.
offset    = -60.
offset_L1 = -670.
process.SiPixelVCal = cms.EDAnalyzer("SiPixelVCalDB",
    BPixParameters = cms.untracked.VPSet(
      cms.PSet(
        layer = cms.uint32(1), # L1
        slope  = cms.double(slope_L1*1.110),
        offset = cms.double(offset_L1),
      ),
      cms.PSet(
        layer = cms.uint32(2), # L2
        slope  = cms.double(slope*1.036),
        offset = cms.double(offset),
      ),
      cms.PSet(
        layer = cms.uint32(3), # L3
        slope  = cms.double(slope*1.023),
        offset = cms.double(offset),
      ),
      cms.PSet(
        layer = cms.uint32(4), # L4
        slope  = cms.double(slope*1.011),
        offset = cms.double(offset),
      ),
    ),
    FPixParameters = cms.untracked.VPSet(
      # See
      #   https://github.com/cms-sw/cmssw/blob/master/DataFormats/SiPixelDetId/src/PixelEndcapName.cc
      # Side: 1=minus, 2=plus
      # Disk: 1, 2, 3
      # Ring: 1=lower, 2=higher
      cms.PSet(
        side = cms.uint32(1), # Rm1 lower
        disk = cms.uint32(1),
        ring = cms.uint32(1),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(1), # Rm1 upper
        disk = cms.uint32(1),
        ring = cms.uint32(2),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(1), # Rm2 lower
        disk = cms.uint32(2),
        ring = cms.uint32(1),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(1), # Rm2 upper
        disk = cms.uint32(2),
        ring = cms.uint32(2),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(1), # Rm3 lower
        disk = cms.uint32(3),
        ring = cms.uint32(1),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(1), # Rm3 upper
        disk = cms.uint32(3),
        ring = cms.uint32(2),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(2), # Rp1 lower
        disk = cms.uint32(1),
        ring = cms.uint32(1),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(2), # Rp1 upper
        disk = cms.uint32(1),
        ring = cms.uint32(2),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(2), # Rp2 lower
        disk = cms.uint32(2),
        ring = cms.uint32(1),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(2), # Rp2 upper
        disk = cms.uint32(2),
        ring = cms.uint32(2),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(2), # Rp3 lower
        disk = cms.uint32(3),
        ring = cms.uint32(1),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
      cms.PSet(
        side = cms.uint32(2), # Rp3 upper
        disk = cms.uint32(3),
        ring = cms.uint32(2),
        slope  = cms.double(slope*1.1275),
        offset = cms.double(offset),
      ),
    ),
    record = cms.untracked.string('SiPixelVCalRcd'),
)

process.SiPixelVCalSim = cms.EDAnalyzer("SiPixelVCalDB",
    record = cms.untracked.string('SiPixelVCalSimRcd'),
)

process.p = cms.Path(
  #process.SiPixelVCalSim*
  process.SiPixelVCal
)
