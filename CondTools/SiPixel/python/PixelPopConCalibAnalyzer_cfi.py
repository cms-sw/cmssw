"""
PixelPopConCalibAnalyzer_cfi.py

Python cfi file for the PixelPopConCalib application.

Original version: Sep 2008, M. Eads
"""

import FWCore.ParameterSet.Config as cms

# set up the EDAnalyzer for the pixel calib config popcon application
PixelPopConCalibAnalyzer = cms.EDAnalyzer('PixelPopConCalibAnalyzer',
                                          # record name - this is defined in CMSSW and shouldn't be changed
                                          record = cms.string('SiPixelCalibConfigurationRcd'),
                                          # choose whether to log the transfer - defaults to true
                                          loggingOn = cms.untracked.bool(True),
                                          # SinceAppendMode controls how the IOV given is interpreted.
                                          # If "true", the sinceIOV parameter will be appended to the IOV.
                                          # So, if sinceIOV is 10, then the IOV written will be from run 10 to infinity
                                          SinceAppendMode = cms.bool(True),
                                          Source = cms.PSet(
                                                            # ?
                                                            firstSince  = cms.untracked.double(300),
                                                            # connectString controls where the calib.dat information is read from
                                                            # to read from a file, it should begin with "file://"
                                                            # to read from the database, it should start with "oracle://"
                                                            # NOTE: READING THE CALIB.DAT FROM THE DATABASE IS NOT CURRENTLY IMPLEMENTED
                                                            connectString = cms.string('file:///afs/cern.ch/user/m/meads/test_calib.dat'),
                                                            # schemaName, viewName, runNumber, and configKeyName are only used when reading from the database
                                                            # (and are not currently used)
                                                            schemaName = cms.string('CMS_PXL_PIXEL_VIEW_OWNER'),
                                                            viewName = cms.string('CONF_KEY_PIXEL)CALIB_V'),
                                                            runNumber = cms.int32(-1),
                                                            configKeyName = cms.string('pixel-config-key-demo2'),
                                                            # sinceIOV (together with SinceAppendMode above) control the IOV that is assigned to this calib configuration object
                                                            sinceIOV = cms.uint32(1)
                                                            )
                                          )

# Use CondDBCOmmon and PoolDBOutputSource to write the calib configuration object to ORCON
from CondCore.DBCommon.CondDBCommon_cfi import *
# connect string that determines to which database the calb config object will be written
CondDBCommon.connect = cms.string('sqlite_file:testExample.db')
# path to the database authentication.xml file
CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBCommon,
    # connection string for the log database
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    # records to put into the database
    toPut = cms.VPSet(cms.PSet(
        # record name - this is set in CMSSW and shouldn't be changed
        record = cms.string('SiPixelCalibConfigurationRcd'),
        # tag name
        tag = cms.string('Pxl_tst_tag1')
    ))
)
