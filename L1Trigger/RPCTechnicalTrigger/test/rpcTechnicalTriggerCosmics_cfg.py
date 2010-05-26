import FWCore.ParameterSet.Config as cms
import os

maxevts   = 1000

# 3_4_X
globaltag = 'STARTUP3X_V14::All'
inputfile  = '/store/relval/CMSSW_3_4_1/RelValCosmics/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/4AE3BADD-B6ED-DE11-A3DF-000423D9863C.root'

# 3_5_X
globaltag = 'START3X_V24::All'
inputfile  = '/store/relval/CMSSW_3_5_4/RelValCosmics/GEN-SIM-RECO/START3X_V24-v1/0004/0442F039-2D2C-DF11-B4B2-00261894380A.root'

# 3_6_X
globaltag = 'START36_V2::All'
inputfile  = '/store/relval/CMSSW_3_6_0_pre3/RelValCosmics/GEN-SIM-RECO/START36_V2-v2/0001/7CA0414B-1C31-DF11-B599-0030487A3232.root'

process   = cms.Process("RPCTechnicalTrigger")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.categories = ['*']
process.MessageLogger.destinations = ['cout']
process.MessageLogger.cout = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG'),
                                                 INFO = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) )

#.. Geometry and Global Tags
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string( globaltag )
process.load("Configuration.StandardSequences.MagneticField_cff")

#.. if cosmics: reconstruction sequence for Cosmics
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(maxevts) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring( inputfile ) )

#..............................................................................................................
#.. EventSetup Configuration
#...

useEventSetup = 0
mytag         = 'test5'
database      = 'sqlite'

if database   == 'sqlite':
    dbconnection = 'sqlite_file:/afs/cern.ch/user/a/aosorio/public/rpcTechnicalTrigger/myrbconfig.db'
elif database == 'oraclerpc':
    dbconnection = 'oracle://devdb10/CMS_RPC_COMMISSIONING'
else:
    dbconnection = ''

if useEventSetup >= 1:

    from CondCore.DBCommon.CondDBCommon_cfi import *

    PoolDBESSource = cms.ESSource("PoolDBESSource",
                                  loadAll = cms.bool(True),
                                  toGet = cms.VPSet(cms.PSet( record = cms.string('RBCBoardSpecsRcd'),
                                                              tag = cms.string(mytag+'a')),
                                                    cms.PSet( record = cms.string('TTUBoardSpecsRcd'),
                                                              tag = cms.string(mytag+'b'))),
                                  DBParameters = cms.PSet( messageLevel = cms.untracked.int32(2),
                                                           authenticationPath = cms.untracked.string('')),
                                  messagelevel = cms.untracked.uint32(2),
                                  connect = cms.string(dbconnection) )

    CondDBCommon.connect = cms.string( dbconnection )

#..............................................................................................................

process.load("L1Trigger.RPCTechnicalTrigger.rpcTechnicalTrigger_cfi")
process.rpcTechnicalTrigger.UseRPCSimLink = cms.untracked.int32(1)
process.rpcTechnicalTrigger.RPCDigiLabel = cms.InputTag("simMuonRPCDigis")

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('rpcttbits.root'),
                               outputCommands = cms.untracked.vstring('drop *','keep L1GtTechnicalTriggerRecord_*_*_*') )

process.p = cms.Path(process.rpcTechnicalTrigger)

process.e = cms.EndPath(process.out)

