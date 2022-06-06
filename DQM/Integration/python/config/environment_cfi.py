from __future__ import print_function
import os
import FWCore.ParameterSet.Config as cms
import configparser as ConfigParser

def loadDQMRunConfigFromFile():
    # try reading the config
    conf_locations = [
        "/etc/dqm_run_config",
        #os.path.expanduser("~/.dqm_run_config"),
        os.path.join(os.curdir, "dqm_run_config"),
    ]
    
    config = ConfigParser.ConfigParser()
    files_read = config.read(conf_locations)

    print("Loaded configuration file from:", files_read)
    return config

# default values, config in file overrides parts of it
# dqmEnv and dqmSaver will configure from this pset
dqmRunConfigDefaults = {
    'userarea': cms.PSet(
        type = cms.untracked.string("userarea"),
        collectorPort = cms.untracked.int32(9190),
        collectorHost = cms.untracked.string('127.0.0.1'),
    ),
    'playback': cms.PSet(
        type = cms.untracked.string("playback"),
        collectorPort = cms.untracked.int32(9090),
        collectorHost = cms.untracked.string('dqm-integration.cms'),
    ),
    'production': cms.PSet(
        type = cms.untracked.string("production"),
        collectorPort = cms.untracked.int32(9090),
        collectorHost = cms.untracked.string('dqm-prod-local.cms'),
    ),
}

# type should be loaded first, to populate the proper defaults
dqmFileConfig = loadDQMRunConfigFromFile()
dqmRunConfigType = "userarea"
if dqmFileConfig.has_option("host", "type"):
    dqmRunConfigType = dqmFileConfig.get("host", "type")
    
isDqmPlayback = cms.PSet( value = cms.untracked.bool( dqmRunConfigType == "playback" ) )
isDqmProduction = cms.PSet( value = cms.untracked.bool( dqmRunConfigType == "production" ) )

dqmRunConfig = dqmRunConfigDefaults[dqmRunConfigType]

# load the options from the config file, if set
if dqmFileConfig.has_option("host", "collectorPort"):
    dqmRunConfig.collectorPort = int(dqmFileConfig.get("host", "collectorPort"))

if dqmFileConfig.has_option("host", "collectorHost"):
    dqmRunConfig.collectorHost = dqmFileConfig.get("host", "collectorHost")

# now start the actual configuration
print("dqmRunConfig:", dqmRunConfig)

from DQMServices.Core.DQMStore_cfi import *

DQM = cms.Service("DQM",
                  debug = cms.untracked.bool(False),
                  publishFrequency = cms.untracked.double(5.0),
                  collectorPort = dqmRunConfig.collectorPort,
                  collectorHost = dqmRunConfig.collectorHost,
                  filter = cms.untracked.string(''),
)

DQMMonitoringService = cms.Service("DQMMonitoringService")

from DQMServices.Components.DQMEventInfo_cfi import *
from DQMServices.FileIO.DQMFileSaverOnline_cfi import *

# upload should be either a directory or a symlink for dqm gui destination
dqmSaver.path = "./upload" 
dqmSaver.tag = "PID%06d" % os.getpid()
dqmSaver.producer = 'DQM'
dqmSaver.backupLumiCount = 15

# Add Protobuf DQM saver
from DQMServices.FileIO.DQMFileSaverPB_cfi import dqmSaver as dqmSaverPB

dqmSaverPB.path = './upload/pb'
dqmSaverPB.tag = 'PID%06d' % os.getpid()
dqmSaverPB.producer = 'DQM'
dqmSaverPB.fakeFilterUnitMode = True
