import FWCore.ParameterSet.Config as cms

process = cms.Process("analyze")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR09_31X_V3P::All'

process.load("Configuration.StandardSequences.GeometryIdeal_cff")

from CondCore.DBCommon.CondDBSetup_cfi import *

import CalibTracker.Configuration.Common.PoolDBESSource_cfi

#Uncomment to change input LA.db file

#process.SiStripLorentzAngle = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#    DBParameters = cms.PSet(messageLevel = cms.untracked.int32(2),
#                            authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#                            ),
#    toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripLorentzAngleRcd'),
#                               tag = cms.string('SiStripLA_TEST_Layers')
#    )),
#    connect = cms.string('sqlite_file:DB_LA_TEST_Layers.db')
#)
#                                      
#process.es_prefer_SiStripLorentzAngle = cms.ESPrefer("PoolDBESSource","SiStripLorentzAngle")

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibTracker.SiStripLorentzAngle.SiStripCalibLorentzAngle_cfi")

process.TrackerGeometricDetESModule = cms.ESProducer("TrackerGeometricDetESModule",fromDDD = cms.bool(True))

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('sistripLACalib')
process.MessageLogger.cerr.enable = False

process.MessageLogger.files.LACalibDebug_Calib =  cms.untracked.PSet(
      threshold = cms.untracked.string('DEBUG'),
      noLineBreaks = cms.untracked.bool(False),
      DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0))
 )

process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:DB_LA_TEST_Modules_Calib.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    connect = cms.string('sqlite_file:DB_LA_TEST_Modules_Calib.db'),
    timetype = cms.untracked.string('runnumber'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')),
    toPut = cms.VPSet(cms.PSet(
	record = cms.string('SiStripLorentzAngleRcd'),
        tag = cms.string('SiStripLA_TEST_Modules_Calib')
    ))
)

process.p = cms.Path(process.sistripLACalib)

process.sistripLACalib.LayerDB = False
process.sistripLACalib.CalibByMC = True
process.sistripLACalib.ModuleFitXMin = -0.5
process.sistripLACalib.ModuleFitXMax = 0.3
process.sistripLACalib.ModuleFit2ITXMin = -0.4
process.sistripLACalib.ModuleFit2ITXMax = 0.2
process.sistripLACalib.p0_guess = -0.1
process.sistripLACalib.p1_guess = 0.5
process.sistripLACalib.p2_guess = 1
process.sistripLACalib.FitCuts_Entries = 1000
process.sistripLACalib.FitCuts_p0 = 10
process.sistripLACalib.FitCuts_p1 = 0.3
process.sistripLACalib.FitCuts_p2 = 1
process.sistripLACalib.FitCuts_chi2 = 10
process.sistripLACalib.FitCuts_ParErr_p0 = 0.001
process.sistripLACalib.GaussFitRange = 0.1

process.sistripLACalib.fileName = 'Summary_CRAFTREPRO_NEWAL.root'
process.sistripLACalib.out_fileName = 'LA_TEST_Calib.root'
process.sistripLACalib.LA_Report = 'LA_Report_TEST_Calib.txt'



