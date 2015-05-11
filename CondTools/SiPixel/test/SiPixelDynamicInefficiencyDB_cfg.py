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

###### DYNAMIC INEFFICIENCY OBJECT ######
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
        layer = cms.uint32(1),
        ladder = cms.uint32(1),
        factor = cms.double(0.978351)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(2),
        factor = cms.double(0.971877)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(3),
        factor = cms.double(0.974283)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(4),
        factor = cms.double(0.969328)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(5),
        factor = cms.double(0.972922)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(6),
        factor = cms.double(0.970964)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(7),
        factor = cms.double(0.975762)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(8),
        factor = cms.double(0.974786)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(9),
        factor = cms.double(0.980244)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(10),
        factor = cms.double(0.978452)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(11),
        factor = cms.double(0.982129)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(12),
        factor = cms.double(0.979737)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(13),
        factor = cms.double(0.984381)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(14),
        factor = cms.double(0.983971)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(15),
        factor = cms.double(0.98186)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(16),
        factor = cms.double(0.983283)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(17),
        factor = cms.double(0.981485)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(18),
        factor = cms.double(0.979753)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(19),
        factor = cms.double(0.980287)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        ladder = cms.uint32(20),
        factor = cms.double(0.975195)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(1),
        factor = cms.double(0.996276)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(2),
        factor = cms.double(0.993354)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(3),
        factor = cms.double(0.993752)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(4),
        factor = cms.double(0.992948)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(5),
        factor = cms.double(0.993871)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(6),
        factor = cms.double(0.992317)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(7),
        factor = cms.double(0.997733)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(8),
        factor = cms.double(0.992516)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(9),
        factor = cms.double(0.992649)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(10),
        factor = cms.double(0.993425)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(11),
        factor = cms.double(0.994065)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(12),
        factor = cms.double(0.993481)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(13),
        factor = cms.double(0.993169)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(14),
        factor = cms.double(0.994223)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(15),
        factor = cms.double(0.992397)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(16),
        factor = cms.double(0.99509)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(17),
        factor = cms.double(0.995177)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(18),
        factor = cms.double(0.995319)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(19),
        factor = cms.double(0.994925)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(20),
        factor = cms.double(0.992933)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(21),
        factor = cms.double(0.994111)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(22),
        factor = cms.double(0.9948)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(23),
        factor = cms.double(0.994711)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(24),
        factor = cms.double(0.994294)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(25),
        factor = cms.double(0.995392)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(26),
        factor = cms.double(0.994229)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(27),
        factor = cms.double(0.994414)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(28),
        factor = cms.double(0.995271)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(29),
        factor = cms.double(0.993585)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(30),
        factor = cms.double(0.995264)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(31),
        factor = cms.double(0.992977)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        ladder = cms.uint32(32),
        factor = cms.double(0.993642)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(1),
        factor = cms.double(0.996206)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(2),
        factor = cms.double(0.998039)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(3),
        factor = cms.double(0.995801)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(4),
        factor = cms.double(0.99665)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(5),
        factor = cms.double(0.996414)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(6),
        factor = cms.double(0.995755)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(7),
        factor = cms.double(0.996518)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(8),
        factor = cms.double(0.995584)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(9),
        factor = cms.double(0.997171)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(10),
        factor = cms.double(0.998056)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(11),
        factor = cms.double(0.99595)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(12),
        factor = cms.double(0.997473)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(13),
        factor = cms.double(0.996858)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(14),
        factor = cms.double(0.996486)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(15),
        factor = cms.double(0.997442)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(16),
        factor = cms.double(0.998002)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(17),
        factor = cms.double(0.995429)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(18),
        factor = cms.double(0.997939)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(19),
        factor = cms.double(0.996896)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(20),
        factor = cms.double(0.997434)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(21),
        factor = cms.double(0.996616)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(22),
        factor = cms.double(0.996439)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(23),
        factor = cms.double(0.996546)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(24),
        factor = cms.double(0.997597)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(25),
        factor = cms.double(0.995435)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(26),
        factor = cms.double(0.996396)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(27),
        factor = cms.double(0.99621)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(28),
        factor = cms.double(0.998316)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(29),
        factor = cms.double(0.998431)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(30),
        factor = cms.double(0.99598)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(31),
        factor = cms.double(0.997063)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(32),
        factor = cms.double(0.996245)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(33),
        factor = cms.double(0.997502)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(34),
        factor = cms.double(0.996254)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(35),
        factor = cms.double(0.997545)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(36),
        factor = cms.double(0.997553)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(37),
        factor = cms.double(0.996722)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(38),
        factor = cms.double(0.996107)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(39),
        factor = cms.double(0.996588)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(40),
        factor = cms.double(0.996277)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(41),
        factor = cms.double(0.99785)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(42),
        factor = cms.double(0.997087)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(43),
        factor = cms.double(0.998139)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        ladder = cms.uint32(44),
        factor = cms.double(0.997139)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        module = cms.uint32(1),
        factor = cms.double(0.953582)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        module = cms.uint32(2),
        factor = cms.double(0.961242)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        module = cms.uint32(3),
        factor = cms.double(0.999371)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        module = cms.uint32(4),
        factor = cms.double(1.00361)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        module = cms.uint32(5),
        factor = cms.double(1.00361)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        module = cms.uint32(6),
        factor = cms.double(0.999371)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        module = cms.uint32(7),
        factor = cms.double(0.961242)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(1),
        module = cms.uint32(8),
        factor = cms.double(0.953582)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        module = cms.uint32(1),
        factor = cms.double(1.00341)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        module = cms.uint32(2),
        factor = cms.double(0.99562)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        module = cms.uint32(3),
        factor = cms.double(0.999792)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        module = cms.uint32(4),
        factor = cms.double(1.00069)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        module = cms.uint32(5),
        factor = cms.double(1.00069)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        module = cms.uint32(6),
        factor = cms.double(0.999792)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        module = cms.uint32(7),
        factor = cms.double(0.99562)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        module = cms.uint32(8),
        factor = cms.double(1.00341)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        module = cms.uint32(1),
        factor = cms.double(1.00039)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        module = cms.uint32(2),
        factor = cms.double(0.998147)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        module = cms.uint32(3),
        factor = cms.double(0.999744)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        module = cms.uint32(4),
        factor = cms.double(1.00006)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        module = cms.uint32(5),
        factor = cms.double(1.00006)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        module = cms.uint32(6),
        factor = cms.double(0.999744)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        module = cms.uint32(7),
        factor = cms.double(0.998147)
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        module = cms.uint32(8),
        factor = cms.double(1.00039)
        ),
      cms.PSet(
        det = cms.string("fpix"),
        factor = cms.double(1)
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
        factor = cms.vdouble(
          1.0181,
          -2.28345e-07,
          -1.30042e-09
          ),
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(2),
        factor = cms.vdouble(
          1.00648,
          -1.28515e-06,
          -1.85915e-10
          ),
        ),
      cms.PSet(
        det = cms.string("bpix"),
        layer = cms.uint32(3),
        factor = cms.vdouble(
          1.0032,
          -1.96206e-08,
          -1.99009e-10
          ),
        ),
      cms.PSet(
        det = cms.string("fpix"),
        factor= cms.vdouble(
          1.0
          ),
        ),
      ),
    theInstLumiScaleFactor = cms.untracked.double(221.95)
    )


process.p = cms.Path(
    process.SiPixelDynamicInefficiency
)

