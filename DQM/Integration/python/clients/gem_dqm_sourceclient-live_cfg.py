import FWCore.ParameterSet.Config as cms
import sys

from Configuration.StandardSequences.Eras import eras
process = cms.Process('GEMDQM', eras.Run3)

unitTest = False
if 'unitTest=True' in sys.argv:
  unitTest=True

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

if unitTest:
  process.load("DQM.Integration.config.unittestinputsource_cfi")
  from DQM.Integration.config.unittestinputsource_cfi import options
else:
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options
  
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "GEM"
process.dqmSaver.tag = "GEM"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "GEM"
process.dqmSaverPB.runNumber = options.runNumber

process.load("DQMServices.Components.DQMProvInfo_cfi")

process.load("EventFilter.GEMRawToDigi.muonGEMDigis_cfi")
process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
process.load("DQM.GEM.GEMDQM_cff")


if (process.runType.getRunType() == process.runType.hi_run):
  process.muonGEMDigis.InputLabel = "rawDataRepacker"

process.muonGEMDigis.useDBEMap = True
process.muonGEMDigis.unPackStatusDigis = True

process.path = cms.Path(
  process.muonGEMDigis *
  process.gemRecHits *
  process.GEMDQM
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver +
  process.dqmSaverPB
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)

process.dqmProvInfo.runType = process.runType.getRunTypeName()

from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
