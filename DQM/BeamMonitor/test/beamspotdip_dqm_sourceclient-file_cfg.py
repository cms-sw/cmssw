from __future__ import print_function
import FWCore.ParameterSet.Config as cms

#
process = cms.Process("BeamSpotDipServer")
process.load("DQMServices.Core.DQM_cfg")

# message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
  debugModules = cms.untracked.vstring('*'),
  cerr = cms.untracked.PSet(
    FwkReport = cms.untracked.PSet(
      optionalPSet = cms.untracked.bool(True),
      reportEvery = cms.untracked.int32(999999999),
    )
  ),
  destinations = cms.untracked.vstring('cerr'),
)

# source
process.source = cms.Source("PoolSource",
  fileNames=cms.untracked.vstring(
    'file:/tmp/sikler/a.root' # lxplus703
  )
)

# module
process.load("DQM.BeamMonitor.BeamSpotDipServer_cff")

process.beamSpotDipServer.verbose = cms.untracked.bool(True)
process.beamSpotDipServer.testing = cms.untracked.bool(True)

process.beamSpotDipServer.sourceFile  = cms.untracked.string(
               "../../../../../BeamFitResults.txt")
process.beamSpotDipServer.sourceFile1 = cms.untracked.string(
               "../../../../../TkStatus.txt")

# process customizations
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

# path
process.p = cms.Path( process.beamSpotDipServer )
