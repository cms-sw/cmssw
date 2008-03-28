# The following comments couldn't be translated into the new config version:

# Database output service

import FWCore.ParameterSet.Config as cms

#
# PixelPopConCalibAnalyzer.cfi
#
# Configuration include file to use PixelPopConCalibAnalyzer
# to do PopCon transfers of calibration configuration objects
#
# M. Eads, Feb 2008
#
# Include needed for PopCon
from CondCore.DBCommon.CondDBCommon_cfi import *
PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBCommon,
    # connection string for the log database
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    # records to put into the database
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelCalibConfigurationRcd'),
        tag = cms.string('Pxl_tst_tag1')
    ))
)

#Common parameters to all subprojects
PixelPopConCalibAnalyzer = cms.EDAnalyzer("PixelPopConCalibAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('SiPixelCalibConfigurationRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        connectString = cms.string('oracle://CMS_PXL_INT2R_LB/CMS_PXL_PRTTYPE_PIXEL_READER'),
        viewName = cms.string('CONF_KEY_PIXEL_CALIB_V'),
        CORAL_AUTH_PATH = cms.untracked.string('./CondTools/SiPixel/data'),
        # "since" IOV number. This is interpreted as a run number and
        # becomes the "since" time for the IOV. Set to 1 to get an
        # infinite IOV
        sinceIOV = cms.uint32(1),
        TNS_ADMIN = cms.untracked.string('/afs/cern.ch/project/oracle/admin'),
        # specify a run number or configuration key name for the OMDS query
        # Set runNumber to -1 to query by config key name
        runNumber = cms.int32(-1),
        # schema and view name used in the OMDS query.
        # put these in the config file in case the database naming change happens
        schemaName = cms.string('CMS_PXL_PIXEL_VIEW_OWNER'),
        configKeyName = cms.string('pixel-config-key-demo1')
    )
)

# connection string for database to populate
CondDBCommon.connect = cms.InputTag("sqlite_file","pop_test.db")

