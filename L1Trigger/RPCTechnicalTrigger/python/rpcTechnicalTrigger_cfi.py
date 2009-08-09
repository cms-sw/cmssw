import FWCore.ParameterSet.Config as cms

trigmode    = 1
usedatabase = 1
mytag       = 'test5'
database    = 'sqlite'

if database == 'sqlite':
    dbconnection = 'sqlite_file:/afs/cern.ch/user/a/aosorio/public/rpcTechnicalTrigger/myrbconfig.db'
else:
    dbconnection = 'oracle://devdb10/CMS_RPC_COMMISSIONING'
    
#...

if usedatabase >= 1:

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

#...

rpcTechnicalTrigger  = cms.EDProducer('RPCTechnicalTrigger',
                                      RPCDigiLabel = cms.InputTag("simMuonRPCDigis"),
                                      UseDatabase = cms.untracked.int32(usedatabase),
                                      BitNumbers=cms.vuint32(24,25,26,27,28,29,30),
                                      BitNames=cms.vstring('L1Tech_rpcBit1',
                                                           'L1Tech_rpcBit2',
                                                           'L1Tech_rpcBit3',
                                                           'L1Tech_rpcBit4',
                                                           'L1Tech_rpcBit5',
                                                           'L1Tech_rpcBit6',
                                                           'L1Tech_rpcBit7',
                                                           ) )


