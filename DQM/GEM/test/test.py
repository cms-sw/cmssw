import FWCore.ParameterSet.Config as cms

import os

from Configuration.Eras.Era_Phase2_cff import Phase2

process = cms.Process('DQMTEST')


process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cerr'),
  cerr = cms.untracked.PSet(
      threshold = cms.untracked.string('WARNING')
  )
)

#process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D35Reco_cff')
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")



process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "GEM"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "GEM"

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    'file:/eos/cms/store/express/Commissioning2018/ExpressCosmics/FEVT/Express-v1/000/310/292/00000/6C23251D-4F18-E811-AEC5-02163E01A41D.root'
  ),
  inputCommands = cms.untracked.vstring(
    'keep *',
    #'keep FEDRawDataCollection_*_*_*'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(8000)
)

process.load("EventFilter.GEMRawToDigi.muonGEMDigis_cfi")
process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
process.load("DQM.GEM.GEMDQM_cff")

process.muonGEMDigis.useDBEMap = True
process.muonGEMDigis.unPackStatusDigis = True

process.GEMDQMStatusDigi.pathOfPrevDQMRoot = "DQM_V0001_GEM_R000030000.root"
process.GEMDQMHarvester.fromFile = "DQM_V0001_GEM_R000020150.root"

############## DB file ################# 
#from CondCore.CondDB.CondDB_cfi import *
#CondDB.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#CondDB.connect = cms.string('sqlite_fip:DQM/GEM/data/GEMeMap.db')
#
#process.GEMCabling = cms.ESSource("PoolDBESSource",
#    CondDB,
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('GEMeMapRcd'),
#        tag = cms.string('GEMeMap_v2')
#    )),
#)
####################################
process.path = cms.Path(
  #process.muonGEMDigis *
  #process.gemRecHits *
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
