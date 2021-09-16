from __future__ import print_function
import FWCore.ParameterSet.Config as cms

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

#process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
#  process.CondDB,
#  toGet = cms.VPSet(
#    cms.PSet(
#      record = cms.string('BeamSpotOnlineLegacyObjectsRcd'),
#      tag = cms.string("BeamSpotOnlineTestLegacy"),
#      refreshTime = cms.uint64(1)
#    ),
#    cms.PSet(
#      record = cms.string('BeamSpotOnlineHLTObjectsRcd'),
#      tag = cms.string("BeamSpotOnlineTestHLT"),
#      refreshTime = cms.uint64(1)
#    )
#  )
#)
#
#process.BeamSpotESProducer = cms.ESProducer("OnlineBeamSpotESProducer")
#process.BeamSpotDBSource.connect = 'frontier://FrontierProd/CMS_CONDITIONS'

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
