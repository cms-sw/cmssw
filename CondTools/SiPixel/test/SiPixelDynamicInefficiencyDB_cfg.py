#import os
import shlex, shutil, getpass
#import subprocess

import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelInclusiveBuilder")
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.destinations = cms.untracked.vstring("cout")
#process.MessageLogger.cout = cms.untracked.PSet(threshold = cms.untracked.string("INFO"))

process.load("Configuration.StandardSequences.MagneticField_cff")

#hptopo
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.autoCond_condDBv2 import autoCond
process.GlobalTag.globaltag = autoCond['run2_design']
print process.GlobalTag.globaltag
process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#to get the user running the process
user = getpass.getuser()

#try:
#    user = os.environ["USER"]
#except KeyError:
#    user = subprocess.call('whoami')
#    # user = commands.getoutput('whoami')
 
#file = "/tmp/" + user + "/SiPixelDynamicInefficiency.db"
file = "siPixelDynamicInefficiency.db"
sqlfile = "sqlite_file:" + file
print '\n-> Uploading as user %s into file %s, i.e. %s\n' % (user, file, sqlfile)


#standard python libraries instead of spawn processes
shutil.move("siPixelDynamicInefficiency.db", "siPixelDynamicInefficiency_old.db")
#subprocess.call(["/bin/cp", "siPixelDynamicInefficiency.db", file])
#subprocess.call(["/bin/mv", "siPixelDynamicInefficiency.db", "siPixelDynamicInefficiency.db"])

##### DATABASE CONNNECTION AND INPUT TAGS ######
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
            record = cms.string('SiPixelDynamicInefficiencyRcd'),
            tag = cms.string('SiPixelDynamicInefficiency_v1')
        ),
                     )
)

###### DYNAMIC INEFFICIENCY OBJECT ###### for 13TeV 25ns case v1
process.SiPixelDynamicInefficiency = cms.EDAnalyzer("SiPixelDynamicInefficiencyDB",
    #in case of PSet
    thePixelGeomFactors = cms.untracked.VPSet(
      cms.PSet(
        det = cms.string("bpix"),
        factor = cms.double(1)
        ),
      cms.PSet(
        det = cms.string("fpix"),
        factor = cms.double(0.999)
        ),
      ),
    theColGeomFactors = cms.untracked.VPSet(
      cms.PSet(
        det = cms.string("bpix"),
        factor = cms.double(1)
        ),
      cms.PSet(
        det = cms.string("fpix"),
        factor = cms.double(0.999)
        ),
      ),
    theChipGeomFactors = cms.untracked.VPSet(
      cms.PSet(
        det = cms.string("bpix"),
        factor = cms.double(1)
        ),
      cms.PSet(
        det = cms.string("fpix"),
        factor = cms.double(0.999)
        ),
      ),
    thePUEfficiency = cms.untracked.VPSet(
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        factor = cms.vdouble(1.00023, -3.18350e-06, 5.08503e-10, -6.79785e-14),
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        factor = cms.vdouble(9.99974e-01, -8.91313e-07, 5.29196e-12, -2.28725e-15 ),
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        factor = cms.vdouble(1.00005, -6.59249e-07, 2.75277e-11, -1.62683e-15 ),
        ),
      cms.PSet(
        det = cms.string("fpix"),
        factor= cms.vdouble(1.0),
        ),
      ),
    theInstLumiScaleFactor = cms.untracked.double(364)
    )


process.p = cms.Path(
    process.SiPixelDynamicInefficiency
)

