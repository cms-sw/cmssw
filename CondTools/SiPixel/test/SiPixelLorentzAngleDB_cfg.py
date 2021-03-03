from __future__ import print_function
#import os
import shlex, shutil, getpass
#import subprocess

import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelInclusiveBuilder")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = dict(enable = True, threshold = "INFO")

process.load("Configuration.StandardSequences.MagneticField_cff")

#hptopo

#process.load("Configuration.StandardSequences.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_design']
print(process.GlobalTag.globaltag)
process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

#process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")

#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

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
 
#file = "/tmp/" + user + "/SiPixelLorentzAngle.db"
file = "siPixelLorentzAngle.db"
sqlfile = "sqlite_file:" + file
print('\n-> Uploading as user %s into file %s, i.e. %s\n' % (user, file, sqlfile))

#standard python libraries instead of spawn processes
shutil.move("siPixelLorentzAngle.db", "siPixelLorentzAngle_old.db")
#subprocess.call(["/bin/cp", "siPixelLorentzAngle.db", file])
#subprocess.call(["/bin/mv", "siPixelLorentzAngle.db", "siPixelLorentzAngle.db"])

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
            record = cms.string('SiPixelLorentzAngleRcd'),
            tag = cms.string('SiPixelLorentzAngle_2015_v2')
            #tag = cms.string('SiPixelLorentzAngle_v1')
        ),
###        cms.PSet(
###            record = cms.string('SiPixelLorentzAngleSimRcd'),
###            tag = cms.string('SiPixelLorentzAngleSim_v1')
###        ),
                     )
)






###### LORENTZ ANGLE OBJECT ######
process.SiPixelLorentzAngle = cms.EDAnalyzer("SiPixelLorentzAngleDB",
    magneticField = cms.double(3.8),
    #in case of PSet
    BPixParameters = cms.untracked.VPSet(
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(1),
            angle = cms.double(0.0862)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(2),
            angle = cms.double(0.0862)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(3),
            angle = cms.double(0.0862)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(4),
            angle = cms.double(0.0862)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(5),
            angle = cms.double(0.0883)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(6),
            angle = cms.double(0.0883)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(7),
            angle = cms.double(0.0883)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(8),
            angle = cms.double(0.0883)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(1),
            angle = cms.double(0.0848)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(2),
            angle = cms.double(0.0848)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(3),
            angle = cms.double(0.0848)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(4),
            angle = cms.double(0.0848)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(5),
            angle = cms.double(0.0892)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(6),
            angle = cms.double(0.0892)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(7),
            angle = cms.double(0.0892)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(8),
            angle = cms.double(0.0892)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(1),
            angle = cms.double(0.0851)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(2),
            angle = cms.double(0.0851)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(3),
            angle = cms.double(0.0851)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(4),
            angle = cms.double(0.0851)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(5),
            angle = cms.double(0.0877)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(6),
            angle = cms.double(0.0877)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(7),
            angle = cms.double(0.0877)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(8),
            angle = cms.double(0.0877)
        ),
    ),
    FPixParameters = cms.untracked.VPSet(
        cms.PSet(
            side = cms.uint32(1),
            disk = cms.uint32(1),
            HVgroup = cms.uint32(1),
            angle = cms.double(0.0714)
        ),
        cms.PSet(
            side = cms.uint32(1),
            disk = cms.uint32(2),
            HVgroup = cms.uint32(1),
            angle = cms.double(0.0714)
        ),
        cms.PSet(
            side = cms.uint32(2),
            disk = cms.uint32(1),
            HVgroup = cms.uint32(1),
            angle = cms.double(0.0713)
        ),
        cms.PSet(
            side = cms.uint32(2),
            disk = cms.uint32(2),
            HVgroup = cms.uint32(1),
            angle = cms.double(0.0713)
        ),
        cms.PSet(
            side = cms.uint32(1),
            disk = cms.uint32(1),
            HVgroup = cms.uint32(2),
            angle = cms.double(0.0643)
        ),
        cms.PSet(
            side = cms.uint32(1),
            disk = cms.uint32(2),
            HVgroup = cms.uint32(2),
            angle = cms.double(0.0643)
        ),
        cms.PSet(
            side = cms.uint32(2),
            disk = cms.uint32(1),
            HVgroup = cms.uint32(2),
            angle = cms.double(0.0643)
        ),
        cms.PSet(
            side = cms.uint32(2),
            disk = cms.uint32(2),
            HVgroup = cms.uint32(2),
            angle = cms.double(0.0643)
        ),
    ),
    ModuleParameters = cms.untracked.VPSet(
        cms.PSet(
            rawid = cms.uint32(302056472),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302056476),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302056212),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302055700),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302055708),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302060308),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302060312),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302059800),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302059548),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302123040),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302122772),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302122776),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302122516),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302122264),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302122272),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302122008),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302121752),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302121496),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302121240),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302121244),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302128920),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302128924),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302129176),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302129180),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302129184),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302128404),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302128408),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302189088),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302188820),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302188832),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302188052),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302187552),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302197784),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302197532),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302197536),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302197016),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302196244),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302195232),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302188824),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302186772),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302186784),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302121992),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302188552),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302187280),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302186768),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302186764),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302186756),
            angle = cms.double(0.0955)
        ),
        cms.PSet(
            rawid = cms.uint32(302197516),
            angle = cms.double(0.0955)
        ),
    ),
    #in case lorentz angle values for bpix should be read from file -> not implemented yet
    useFile = cms.bool(False),
    record = cms.untracked.string('SiPixelLorentzAngleRcd'),  
    fileName = cms.string('lorentzFit.txt')	
)

process.SiPixelLorentzAngleSim = cms.EDAnalyzer("SiPixelLorentzAngleDB",
    magneticField = cms.double(3.8),
    #in case lorentz angle values for bpix should be read from file -> not implemented yet
    useFile = cms.bool(False),
    record = cms.untracked.string('SiPixelLorentzAngleSimRcd'),
    fileName = cms.string('lorentzFit.txt')	
)




process.p = cms.Path(
#    process.SiPixelLorentzAngleSim*
    process.SiPixelLorentzAngle
    )

