"""
PixelPopConCalibChecker_cfg.py

Python configuration file to run PixelPopConCalibChecker EDanalyzer, which 
checks calib configuration objects transferred into the database.

M. Eads
Aug 2008
"""

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# define the EmptyIOVSource
# firstRun and lastRun should both be set to the run number you want to check
process.source = cms.Source('EmptyIOVSource',
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
)

from CondTools.SiPixel.SiPixelCalibConfiguration_cfi import *
# select the database from which to read the calib configuration object
sipixelcalib_essource.connect = 'sqlite_file:/path/to/testExample.db'

sipixelcalib_essource.toGet = cms.VPSet(cms.PSet(# record is specified in CMSSW and shouldn't be changed
                                                 record = cms.string('SiPixelCalibConfigurationRcd'),
                                                 # change the tag to the tag used when loading the calib configuration object
                                                 tag = cms.string('mytest')
                                                 )
                                        )
process.sipixelcalib_essource = sipixelcalib_essource


process.demo = cms.EDAnalyzer('PixelPopConCalibChecker',
                              # filename is the path to the calib.dat file you want to compare to the calib configuration object in the database
                              filename = cms.string('/afs/cern.ch/user/m/meads/test_calib.dat'),
                              # messageLevel controls the verbosity of the output. 2 (or larger) spits out everything
                              messageLevel = cms.untracked.int32(2)
                              )


process.p = cms.Path(process.demo)
