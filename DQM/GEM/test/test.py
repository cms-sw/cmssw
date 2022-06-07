import FWCore.ParameterSet.Config as cms

import os

from Configuration.Eras.Era_Run3_cff import Run3
#from Configuration.StandardSequences.Eras import eras
process = cms.Process('DQM', Run3)


process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cerr'),
  cerr = cms.untracked.PSet(
      threshold = cms.untracked.string('WARNING')
  )
)

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(None, 'auto:phase1_2022_cosmics', '')
#process.load("DQM.Integration.config.FrontierCondition_GT_cfi")


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
import sys
AddFile = lambda l: [ s if s.startswith("root://") else "file:" + s for s in l ]
listCand = [ s for s in sys.argv if s.endswith(".root") ]
listSrc  = AddFile(listCand)
listCand = [ s for s in sys.argv if s.endswith(".txt") ]
for s in listCand: 
  with open(s) as fSrc: listSrc += AddFile(fSrc.read().splitlines())
if len(listSrc) > 0: process.source.fileNames = cms.untracked.vstring(*listSrc)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.load("EventFilter.GEMRawToDigi.muonGEMDigis_cfi")
process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
process.load("DQM.GEM.GEMDQM_cff")

process.muonGEMDigis.useDBEMap = True
#process.muonGEMDigis.unPackStatusDigis = True  # DEFAULT
process.muonGEMDigis.keepDAQStatus = True  # DEFAULT
#process.muonGEMDigis.unPackStatusDigis = False

#process.GEMDQMStatusDigi.pathOfPrevDQMRoot = "DQM_V0001_GEM_R000030000.root"
#process.GEMDQMHarvester.fromFile = "DQM_V0001_GEM_R000020150.root"

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
  process.muonGEMDigis *  # DEFAULT
  process.gemRecHits *    # DEFAULT
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
