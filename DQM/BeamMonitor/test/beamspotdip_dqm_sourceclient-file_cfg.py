from __future__ import print_function
import FWCore.ParameterSet.Config as cms

# copy log4cplus.properties from >script directory< to >local<
import sys
import os
from shutil import copy
configFile = os.path.dirname(sys.argv[0]) + "/log4cplus.properties"
print("copying " + configFile + " to local")
copy(configFile,".")

#
process = cms.Process("BeamSpotDipServer")
process.load("DQMServices.Core.DQM_cfg")

# message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
       limit = cms.untracked.int32(1000)
    ),
    BeamSpotDipServer = cms.untracked.PSet(
        limit = cms.untracked.int32(1000)
    )
)

# source
process.source = cms.Source("PoolSource",
  fileNames=cms.untracked.vstring(
    'file:/tmp/sikler/b.root' # lxplus7101
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(100)
)

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

process.beamSpotDipServer.verbose = True
process.beamSpotDipServer.testing = True

process.beamSpotDipServer.readFromNFS = True
process.beamSpotDipServer.sourceFile  = "../../../../../BeamFitResults.txt"
process.beamSpotDipServer.sourceFile1 = "../../../../../TkStatus.txt"

# process customizations
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

# path
process.p = cms.Path( process.beamSpotDipServer )
