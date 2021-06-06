from __future__ import print_function
import os
import shlex, subprocess
import shutil, getpass

import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelInclusiveBuilder")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = dict(enable = True, threshold = "INFO")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

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

try:
    user = os.environ["USER"]
except KeyError:
    # user = commands.getoutput('whoami')
    # user = subprocess.call('whoami')
    # faster, "cheaper" (in terms of resources), and more secure
    user = getpass.getuser()
 
#file = "/tmp/" + user + "/prova.db"
file = "prova.db"
sqlfile = "sqlite_file:" + file
print('\n-> Uploading as user %s into file %s, i.e. %s\n' % (user, file, sqlfile))

#subprocess.call(["/bin/cp", "prova.db", file])
#subprocess.call(["/bin/mv", "prova.db", "prova_old.db"])
#faster as it doesn't spawn a process
shutil.move("prova.db", "prova_old.db")


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
#        cms.PSet(
#            record = cms.string('SiPixelFedCablingMapRcd'),
#            tag = cms.string('SiPixelFedCablingMap_v14')
#        ), 
        cms.PSet(
            record = cms.string('SiPixelLorentzAngleRcd'),
            tag = cms.string('SiPixelLorentzAngle_v01')
        ),
###        cms.PSet(
###            record = cms.string('SiPixelLorentzAngleSimRcd'),
###            tag = cms.string('SiPixelLorentzAngleSim_v01')
###        ),
#        cms.PSet(
#            record = cms.string('SiPixelTemplateDBObjectRcd'),
#            tag = cms.string('SiPixelTemplateDBObject')
#        ),
#        cms.PSet(
#           record = cms.string('SiPixelQualityFromDbRcd'),
#           tag = cms.string('SiPixelQuality_test')
#        ),
#        cms.PSet(
#            record = cms.string('SiPixelGainCalibrationOfflineRcd'),
#            tag = cms.string('SiPixelGainCalibration_TBuffer_const')
#        ), 
#        cms.PSet(
#            record = cms.string('SiPixelGainCalibrationForHLTRcd'),
#            tag = cms.string('SiPixelGainCalibration_TBuffer_hlt_const')
#        ),
#        cms.PSet(
#            record = cms.string('SiPixelGainCalibrationOfflineSimRcd'),
#            tag = cms.string('SiPixelGainCalibrationSim_TBuffer_const_new')
#        ), 
#        cms.PSet(
#            record = cms.string('SiPixelGainCalibrationForHLTSimRcd'),
#            tag = cms.string('SiPixelGainCalibrationSim_TBuffer_hlt_const')
#        )
                     )
)








###### TEMPLATE OBJECT UPLOADER ######
MagFieldValue = 3.8
if ( MagFieldValue==0 ):
    MagFieldString = '0'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0022.out",
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0023.out")
    theDetIds      = cms.vuint32( 1, 2) # 0 is for all, 1 is Barrel, 2 is EndCap
    theTemplateIds = cms.vuint32(22,23)
elif(MagFieldValue==4):
    MagFieldString = '4'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0018.out",
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0019.out")
    theDetIds      = cms.vuint32( 1, 2)
    theTemplateIds = cms.vuint32(18,19)
elif(MagFieldValue==3.8 or MagFieldValue==38):
    MagFieldString = '38'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0020.out",
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0021.out")
    theDetIds      = cms.vuint32( 1, 2)
    theTemplateIds = cms.vuint32(20,21)
elif(MagFieldValue==2):
    MagFieldString = '2'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0030.out",
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0031.out")
    theDetIds      = cms.vuint32( 1, 2)
    theTemplateIds = cms.vuint32(30,31)
elif(MagFieldValue==3):
    MagFieldString = '3'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0032.out",
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0033.out")
    theDetIds      = cms.vuint32( 1, 2)
    theTemplateIds = cms.vuint32(32,33)
elif(MagFieldValue==3.5 or MagFieldValue==35):
    MagFieldString = '35'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0034.out",
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0035.out")
    theDetIds      = cms.vuint32( 1, 2)
    theTemplateIds = cms.vuint32(34,35)

version = "v2"
template_base = 'SiPixelTemplateDBObject' + MagFieldString + 'T'
print('\nUploading %s%s with record SiPixelTemplateDBObjectRcd in file siPixelTemplates%sT.db\n' % (template_base,version,MagFieldString))

process.TemplateUploader = cms.EDAnalyzer("SiPixelTemplateDBObjectUploader",
                                          siPixelTemplateCalibrations = files_to_upload,
                                          theTemplateBaseString = cms.string(template_base),
                                          Version = cms.double(3.0),
                                          MagField = cms.double(MagFieldValue),
                                          detIds = theDetIds,
                                          templateIds = theTemplateIds
)




###### QUALITY OBJECT MAKER #######
process.QualityObjectMaker = cms.EDAnalyzer("SiPixelBadModuleByHandBuilder",
    BadModuleList = cms.untracked.VPSet(cms.PSet(
        errortype = cms.string('whole'),
        detid = cms.uint32(302197784)
         ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(302195232)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344014348)
        )),
    Record = cms.string('SiPixelQualityFromDbRcd'),
    SinceAppendMode = cms.bool(True),
    IOVMode = cms.string('Run'),
    printDebug = cms.untracked.bool(True),
    doStoreOnDB = cms.bool(True)

)



##### CABLE MAP OBJECT ######
process.PixelToLNKAssociateFromAsciiESProducer = cms.ESProducer("PixelToLNKAssociateFromAsciiESProducer",
    fileName = cms.string('pixelToLNK.ascii')
)


process.MapWriter = cms.EDAnalyzer("SiPixelFedCablingMapWriter",
    record = cms.string('SiPixelFedCablingMapRcd'),
    associator = cms.untracked.string('PixelToLNKAssociateFromAscii')
)



###### LORENTZ ANGLE OBJECT ######
process.SiPixelLorentzAngle = cms.EDAnalyzer("SiPixelLorentzAngleDB",
    magneticField = cms.double(3.8),
    #in case of PSet
    BPixParameters = cms.untracked.VPSet(
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(1),
            angle = cms.double(0.09103)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(2),
            angle = cms.double(0.09103)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(3),
            angle = cms.double(0.09103)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(4),
            angle = cms.double(0.09103)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(5),
            angle = cms.double(0.09574)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(6),
            angle = cms.double(0.09574)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(7),
            angle = cms.double(0.09574)
        ),
        cms.PSet(
            layer = cms.uint32(1),
            module = cms.uint32(8),
            angle = cms.double(0.09574)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(1),
            angle = cms.double(0.09415)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(2),
            angle = cms.double(0.09415)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(3),
            angle = cms.double(0.09415)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(4),
            angle = cms.double(0.09415)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(5),
            angle = cms.double(0.09955)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(6),
            angle = cms.double(0.09955)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(7),
            angle = cms.double(0.09955)
        ),
        cms.PSet(
            layer = cms.uint32(2),
            module = cms.uint32(8),
            angle = cms.double(0.09955)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(1),
            angle = cms.double(0.09541)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(2),
            angle = cms.double(0.09541)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(3),
            angle = cms.double(0.09541)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(4),
            angle = cms.double(0.09541)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(5),
            angle = cms.double(0.10121)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(6),
            angle = cms.double(0.10121)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(7),
            angle = cms.double(0.10121)
        ),
        cms.PSet(
            layer = cms.uint32(3),
            module = cms.uint32(8),
            angle = cms.double(0.10121)
        ),
    ),
    FPixParameters = cms.untracked.VPSet(
        cms.PSet(
            side = cms.uint32(1),
            disk = cms.uint32(1),
            HVgroup = cms.uint32(1),
            angle = cms.double(0.06404)
        ),
        cms.PSet(
            side = cms.uint32(1),
            disk = cms.uint32(2),
            HVgroup = cms.uint32(1),
            angle = cms.double(0.06404)
        ),
        cms.PSet(
            side = cms.uint32(2),
            disk = cms.uint32(1),
            HVgroup = cms.uint32(1),
            angle = cms.double(0.06404)
        ),
        cms.PSet(
            side = cms.uint32(2),
            disk = cms.uint32(2),
            HVgroup = cms.uint32(1),
            angle = cms.double(0.06404)
        ),
        cms.PSet(
            side = cms.uint32(1),
            disk = cms.uint32(1),
            HVgroup = cms.uint32(2),
            angle = cms.double(0.06404)
        ),
        cms.PSet(
            side = cms.uint32(1),
            disk = cms.uint32(2),
            HVgroup = cms.uint32(2),
            angle = cms.double(0.06404)
        ),
        cms.PSet(
            side = cms.uint32(2),
            disk = cms.uint32(1),
            HVgroup = cms.uint32(2),
            angle = cms.double(0.06404)
        ),
        cms.PSet(
            side = cms.uint32(2),
            disk = cms.uint32(2),
            HVgroup = cms.uint32(2),
            angle = cms.double(0.06404)
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

###### OFFLINE GAIN OBJECT ######
process.SiPixelCondObjOfflineBuilder = cms.EDAnalyzer("SiPixelCondObjOfflineBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.2),
    rmsPed = cms.double(0.),
# separate input for the FPIX. If not entered the default values are used.
    rmsGainFPix = cms.untracked.double(0.),
    meanGainFPix = cms.untracked.double(2.8),
    meanPedFPix = cms.untracked.double(28.2),
    rmsPedFPix = cms.untracked.double(0.),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationOfflineRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)

process.SiPixelCondObjOfflineBuilderSim = cms.EDAnalyzer("SiPixelCondObjOfflineBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.2),
    rmsPed = cms.double(0.),
# separate input for the FPIX. If not entered the default values are used.
    rmsGainFPix = cms.untracked.double(0.),
    meanGainFPix = cms.untracked.double(2.8),
    meanPedFPix = cms.untracked.double(28.2),
    rmsPedFPix = cms.untracked.double(0.),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationOfflineSimRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)


##### HLT GAIN OBJECT #####
process.SiPixelCondObjForHLTBuilder = cms.EDAnalyzer("SiPixelCondObjForHLTBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.0),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.0),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationForHLTRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    rmsPed = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)

process.SiPixelCondObjForHLTBuilderSim = cms.EDAnalyzer("SiPixelCondObjForHLTBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.0),
    rmsPed = cms.double(0.0),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.0),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationForHLTSimRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)


process.p = cms.Path(
#    process.MapWriter*
#    process.SiPixelCondObjOfflineBuilder*
#    process.SiPixelCondObjForHLTBuilder*
#    process.TemplateUploader*
#    process.QualityObjectMaker*
#    process.SiPixelLorentzAngleSim*
#    process.SiPixelCondObjForHLTBuilderSim*
#    process.SiPixelCondObjOfflineBuilderSim*
    process.SiPixelLorentzAngle
    )

