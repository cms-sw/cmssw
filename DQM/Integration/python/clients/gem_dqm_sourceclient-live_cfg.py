import FWCore.ParameterSet.Config as cms

process = cms.Process('GEMDQM')

process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "GEM"
process.dqmSaver.tag = "GEM"

process.load("DQM.Integration.config.inputsource_cfi")

process.load("DQMServices.Components.DQMProvInfo_cfi")

process.load("EventFilter.GEMRawToDigi.muonGEMDigis_cfi")
process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
process.load("DQM.GEM.GEMDQM_cff")


if (process.runType.getRunType() == process.runType.hi_run):
  process.muonGEMDigis.InputLabel = cms.InputTag("rawDataRepacker")                                                                                                                                          

process.muonGEMDigis.useDBEMap = True
process.muonGEMDigis.unPackStatusDigis = True

process.path = cms.Path(
  process.muonGEMDigis *
  process.gemRecHits *
  process.GEMDQM
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)

process.dqmProvInfo.runType = process.runType.getRunTypeName()

from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
