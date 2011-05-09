import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCPVT")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'MC_31X_V1::All'
process.GlobalTag.globaltag = 'START3X_V18::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
# output database (in this case local sqlite file)
process.CondDBCommon.connect = 'sqlite_file:dbfile.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
        process.CondDBCommon,
        timetype = cms.untracked.string('runnumber'),
        logconnect= cms.untracked.string('sqlite_file:log.db'),
        toPut = cms.VPSet(cms.PSet(
            record = cms.string('RPCDQMObjectRcd'),
            tag = cms.string('RPC_test')
            ))
        )

process.source = cms.Source("EmptySource",
                           firstRun = cms.untracked.uint32(142128) 
                         )

process.readMeFromFile = cms.EDAnalyzer("ReadMeFromFile",
             #InputFile = cms.untracked.string('DQMPVT.root')
             InputFile = cms.untracked.string('dqmdata/DQM_V0001_R000142128__Mu__Run2010A-Sep17ReReco_v2__DQM.root')
             )

process.rpcpopcon = cms.EDAnalyzer('RPCDBPopConAnalyzer',
             record = cms.string('RPCDQMObjectRcd'),
             Source=cms.PSet(
               IOVRun = cms.untracked.uint32(142128)
             ),
             RecHitTypeFolder = cms.untracked.string('RecHits'),
)

#process.p = cms.Path(process.dqmpvt)
process.p = cms.Path(process.readMeFromFile*process.rpcpopcon)
#process.p = cms.Path(process.rpcdbclient)

############# Message Logger ####################
process.MessageLogger = cms.Service("MessageLogger",
     debugModules = cms.untracked.vstring('*'),
     cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
     destinations = cms.untracked.vstring('cout')
 )
