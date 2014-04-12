"""
testPixelPopConCalibAnalyzer_cfg.pg

Python configuration file to run the pixel calib configuration popcon application

Original version: Sep 2008, M. Eads
"""

import FWCore.ParameterSet.Config as cms

process = cms.Process("testPixelPopConCalibAnalyzer")
# load the cfi for the PixelPopConCalib application
process.load("CondTools.SiPixel.PixelPopConCalibAnalyzer_cfi")

# define the source for the CMSSW process
process.source = cms.Source('EmptyIOVSource',
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

# change the location the calib.dat data is read from
#process.PixelPopConCalibAnalyzer.Source.connectString = 'file:///path/to/calib.dat'

# change the run number used to set the IOV
# by default, if sinceIOV is N, then the IOV will be from run N to infinity
#process.PixelPopConCalibAnalyzer.Source.sinceIOV = 2

# change the logging db used
#process.PoolDBOutputService.logconnect = 'sqlite_file:my_logging.db'

# change the tag name used
#process.PoolDBOutputService.toPut[0].tag = 'my_tagname'

# change the database that the calib config object is written to 
#process.PoolDBOutputService.connect = 'sqlite_file:my.db' 

process.p = cms.Path(process.PixelPopConCalibAnalyzer)
