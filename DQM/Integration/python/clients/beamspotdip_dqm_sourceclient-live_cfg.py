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

process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

process.GlobalTag.toGet = cms.VPSet(
  cms.PSet(
    record = cms.string("BeamSpotOnlineLegacyObjectsRcd"),
    refreshTime = cms.uint64(1)
  ),
)

# module
process.load("DQM.BeamMonitor.BeamSpotDipServer_cff")

# process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

# path
process.p = cms.Path( process.beamSpotDipServer )
