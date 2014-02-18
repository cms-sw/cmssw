
import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCDQM")

############# Source File #################

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/2EB74417-51AF-DF11-8773-001617E30D00.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/4420AAFC-3BAF-DF11-8457-0019B9F72BAA.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/58BD28DA-4CAF-DF11-912A-001617E30D00.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/5E4D5A83-54AF-DF11-9580-001D09F24934.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/6C72A6E0-40AF-DF11-B657-001D09F252F3.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/7A94925E-4CAF-DF11-9C5E-001D09F2960F.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/8A10CC2F-47AF-DF11-BB2F-0030487A17B8.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/8CAFACF6-49AF-DF11-861A-003048F118AC.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/9A3E9521-53AF-DF11-84AC-001D09F24934.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/BEE0AAB1-37AF-DF11-9FB8-0030487C60AE.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/C4EF4794-48AF-DF11-AC6B-003048F117EA.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/D2381722-53AF-DF11-8D77-001D09F2906A.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/E4185ADF-4CAF-DF11-BA63-0030487CD906.root',
                                                              # '/store/data/Run2010A/Mu/RECO/v4/000/143/727/FE2C2F5F-4CAF-DF11-AA49-0030487A3C9A.root',
     'file:/build/piet/Upgrade/Eta_2p4_Releases/CMSSW_6_2_0_SLHC5/src/MyCmsDriverCommands/SingleMuPt100_1p6_2p4_cfi_RECO.root'
     )
                            )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

################ Condition #################
process.load("CondCore.DBCommon.CondDBSetup_cfi")

############ Geometry ######################
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
# process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.CMSCommonData.cmsExtendedGeometry2023RPCEtaUpscopeXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

############ RAW to DIGI ###################
## process.rpcunpacker = cms.EDFilter("RPCUnpackingModule",
##     InputLabel = cms.InputTag("source"),
##     doSynchro = cms.bool(False)
## )
process.load("EventFilter.RPCRawToDigi.RPCFrontierCabling_cfi")

############ RecHits #######################
process.load("RecoLocalMuon.RPCRecHit.rpcRecHits_cfi")
#process.rpcRecHits.rpcDigiLabel ='rpcunpacker'
process.rpcRecHits.rpcDigiLabel = 'muonRPCDigis'
process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

################ DQM #######################
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'RPC'
process.dqmSaver.convention = 'Online'

############# RPC Monitor Digi #############
process.load("DQM.RPCMonitorDigi.RPCDigiMonitoring_cfi")

########### RPC RecHit Probability #########
process.load("DQM.RPCMonitorDigi.RPCRecHitProbability_cfi")

################### FED ####################
process.load("DQM.RPCMonitorClient.RPCMonitorRaw_cfi")
process.load("DQM.RPCMonitorClient.RPCFEDIntegrity_cfi")
process.rpcFEDIntegrity.RPCRawCountsInputTag = 'provaDiNoCrash'
process.load("DQM.RPCMonitorClient.RPCMonitorLinkSynchro_cfi")

############### Output Module ##############
process.out = cms.OutputModule("PoolOutputModule",
   # fileName = cms.untracked.string('/tmp/cimmino/RPCDQM.root'),
   fileName = cms.untracked.string('/tmp/piet/RPCDQM.root'),
   outputCommands = cms.untracked.vstring("keep *")
)

############# Message Logger ###############
process.MessageLogger = cms.Service("MessageLogger",
     debugModules = cms.untracked.vstring('*'),
     cout = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG')),
     destinations = cms.untracked.vstring('cout')
)

############ Memory check ##################
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
        ignoreTotal = cms.untracked.int32(1) ## default is one
) 

################## Timing #################
process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))

############# Path ########################
process.p = cms.Path(process.rpcdigidqm*process.rpcrechitprobability*process.dqmEnv*process.dqmSaver)

process.e = cms.EndPath(process.out)


