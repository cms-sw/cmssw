#! /usr/bin/env cmsRun
# Author: Izaak Neutelings (March 2020)
from __future__ import print_function
#import os
import os, shlex, shutil, getpass
#import subprocess
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

# SETTINGS
run       = 313000 # select the geometry for Phase-I pixels
era       = eras.Run2_2017 
verbose   = False #or True
threshold = 'INFO' if verbose else 'WARNING'
print(">>> run = %s"%run)

# LOAD PROCESS
process = cms.Process("SiPixelVCalDB",era)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
#process.load("Configuration.StandardSequences.GeometryIdeal_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")
#process.load("CondCore.DBCommon.CondDBCommon_cfi") # deprecated
process.load("CondCore.CondDB.CondDB_cfi")

# GLOBAL TAG
from Configuration.AlCa.GlobalTag import GlobalTag
#from Configuration.AlCa.autoCond import autoCond
#from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = "auto:run2_data" 
#process.GlobalTag.globaltag = autoCond['run2_design']
# In case you of conditions missing, or if you want to test a specific GT
#process.GlobalTag.globaltag = 'PRE_DES72_V6'
process.GlobalTag = GlobalTag(process.GlobalTag,'auto:run2_data','')
#process.GlobalTag = GlobalTag(process.GlobalTag,autoCond['run2_design'],'')
#process.GlobalTag.globaltag = autoCond['run2_design']
print(">>> globaltag = '%s'"%(process.GlobalTag.globaltag))

# BASIC SETTING
process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(run),
    lastValue = cms.uint64(run),
    #firstRun = cms.untracked.uint32(run),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1),
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# MESSAGER
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = dict(enable = True, threshold = threshold)

# BACK UP DATABASE FILE
user = getpass.getuser()
#try:
#    user = os.environ["USER"]
#except KeyError:
#    user = subprocess.call('whoami')
#    # user = commands.getoutput('whoami') 
#file = "/tmp/" + user + "/SiPixelVCal.db"
file = "siPixelVCal.db"
sqlfile = "sqlite_file:" + file
print(">>> Uploading as user %s into file %s, i.e. %s"%(user,file,sqlfile))
if os.path.exists("siPixelVCal.db"):
  oldfile = file.replace(".db","_old.db")
  print(">>> Backing up locally existing '%s' -> '%s'"%(file,oldfile))
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

# VCAL -> NUMBER OF ELECTRONS DB OBJECT
# https://github.com/cms-analysis/DPGAnalysis-SiPixelTools/blob/1232a8c0ef3abe7b78c757887138089706e0499a/GainCalibration/test/vcal-irradiation-factors.txt
# https://github.com/cms-sw/cmssw/blob/master/DataFormats/SiPixelDetId/src/PixelEndcapName.cc
slope      = 47.
slope_L1   = 50.
offset     = -60.
offset_L1  = -670.
corrs_bpix = { 1: 1.110, 2: 1.036, 3: 1.023, 4: 1.011 }
corrs_fpix = { 1: 1.1275, 2: 1.1275, 3: 1.1275 }
layers     = [1,2,3,4]
nladders   = { 1: 12, 2: 28, 3: 44, 4: 64, }
sides      = [1,2]   # 1=minus, 2=plus
disks      = [1,2,3] # 1, 2, 3
rings      = [1,2]   # 1=lower, 2=upper
bpixpars   = cms.untracked.VPSet()
fpixpars   = cms.untracked.VPSet()
print(">>> %8s %8s %10s %10s %10s"%('layer','ladder','slope','offset','corr'))
for layer in layers:
  for ladder in range(1,nladders[layer]+1):
    corr     = corrs_bpix[layer]
    slope_   = (slope_L1 if layer==1 else slope)*corr
    offset_  = (offset_L1 if layer==1 else offset)
    print(">>> %8d %8d %10.4f %10.3f %10.4f"%(layer,ladder,slope_,offset_,corr))
    bpixpars.append(cms.PSet(
      layer  = cms.int32(layer),
      ladder = cms.int32(ladder),
      slope  = cms.double(slope_),
      offset = cms.double(offset_),
    ))
print(">>> %8s %8s %8s %10s %10s %10s"%('side','disk','ring','slope','offset','corr'))
for side in sides:
  for disk in disks:
    for ring in rings:
      corr     = corrs_fpix[ring]
      slope_   = slope*corr
      offset_  = offset
      print(">>> %8d %8d %8d %10.4f %10.3f %10.4f"%(side,disk,ring,slope_,offset_,corr))
      fpixpars.append(cms.PSet(
        side   = cms.int32(side),
        disk   = cms.int32(disk),
        ring   = cms.int32(ring),
        slope  = cms.double(slope_),
        offset = cms.double(offset_),
      ))

# DB CREATOR
process.SiPixelVCal = cms.EDAnalyzer("SiPixelVCalDB",
    BPixParameters = bpixpars,
    FPixParameters = fpixpars,
    record = cms.untracked.string('SiPixelVCalRcd'),
)
process.p = cms.Path(process.SiPixelVCal)
