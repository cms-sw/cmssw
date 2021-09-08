from __future__ import print_function
import FWCore.ParameterSet.Config as cms

#
process = cms.Process("BeamSpotDipServer")
process.load("DQMServices.Core.DQM_cfg")

# input
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")
# for testing in lxplus
# process.load("DQM.Integration.config.fileinputsource_cfi")

# beamspot from database
process.load("CondCore.CondDB.CondDB_cfi")

process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
  process.CondDB,
  toGet = cms.VPSet(
    cms.PSet(
      record = cms.string('BeamSpotOnlineLegacyObjectsRcd'),
      tag = cms.string("BeamSpotOnlineTestLegacy"),
      refreshTime = cms.uint64(1)
    ),
    cms.PSet(
      record = cms.string('BeamSpotOnlineHLTObjectsRcd'),
      tag = cms.string("BeamSpotOnlineTestHLT"),
      refreshTime = cms.uint64(1)
    )
  )
)

process.BeamSpotESProducer = cms.ESProducer("OnlineBeamSpotESProducer")
process.BeamSpotDBSource.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')

# module
process.load("DQM.BeamMonitor.BeamSpotDipServer_cff")

# process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

# path
process.p = cms.Path( process.beamSpotDipServer )
