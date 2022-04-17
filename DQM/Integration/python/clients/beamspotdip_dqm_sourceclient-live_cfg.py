from __future__ import print_function
import FWCore.ParameterSet.Config as cms

# copy log4cplus.properties from >script directory< to >local<
import sys
import os
from shutil import copy
configFile = os.path.dirname(sys.argv[1]) + "/log4cplus.properties"
print("copying " + configFile + " to local")
copy(configFile,".")

#
from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("BeamSpotDipServer", Run3)
process.load("DQMServices.Core.DQM_cfg")

# message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
       limit = cms.untracked.int32(100000)
    ),
    BeamSpotDipServer = cms.untracked.PSet(
        limit = cms.untracked.int32(100000)
    )
)

# input
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")
# for testing in lxplus
# process.load("DQM.Integration.config.fileinputsource_cfi")

# beamspot from database
process.load("CondCore.CondDB.CondDB_cfi")

process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

process.GlobalTag.toGet = cms.VPSet(
  cms.PSet(
    record = cms.string("BeamSpotOnlineLegacyObjectsRcd"),
    refreshTime = cms.uint64(1)
  ),
)

# module
process.load("DQM.BeamMonitor.BeamSpotDipServer_cff")

process.beamSpotDipServer.verbose = cms.untracked.bool(True)

# process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

# monitoring
process.DQMMonitoringService = cms.Service("DQMMonitoringService")

# path
process.p = cms.Path( process.beamSpotDipServer )
print("Final Source settings:", process.source)

