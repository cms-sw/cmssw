
import FWCore.ParameterSet.Config as cms

## Use RECO Muons flag
useMuons = True
isOfflineDQM = True

process = cms.Process("RPCDQM")

############# Source File #################
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/c/cimmino/Mu/165993/00472EF2-7F8B-E011-9D6A-001D09F2915A.root'))
                            
## process.source = cms.Source("PoolSource",
##                             fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/2EB74417-51AF-DF11-8773-001617E30D00.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/4420AAFC-3BAF-DF11-8457-0019B9F72BAA.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/58BD28DA-4CAF-DF11-912A-001617E30D00.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/5E4D5A83-54AF-DF11-9580-001D09F24934.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/6C72A6E0-40AF-DF11-B657-001D09F252F3.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/7A94925E-4CAF-DF11-9C5E-001D09F2960F.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/8A10CC2F-47AF-DF11-BB2F-0030487A17B8.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/8CAFACF6-49AF-DF11-861A-003048F118AC.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/9A3E9521-53AF-DF11-84AC-001D09F24934.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/BEE0AAB1-37AF-DF11-9FB8-0030487C60AE.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/C4EF4794-48AF-DF11-AC6B-003048F117EA.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/D2381722-53AF-DF11-8D77-001D09F2906A.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/E4185ADF-4CAF-DF11-BA63-0030487CD906.root',
##                                                               'rfio:/castor/cern.ch/user/c/cimmino/Mu/143727/FE2C2F5F-4CAF-DF11-AA49-0030487A3C9A.root')
##                             )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

################ Condition #################
process.load("CondCore.DBCommon.CondDBSetup_cfi")
#process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
#process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)

############ Geometry ######################
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
#process.load("Geometry.CSCGeometry.cscGeometry_cfi")

#process.load("Geometry.DTGeometry.dtGeometry_cfi")
#process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

############ RAW to DIGI ###################
## process.rpcunpacker = cms.EDFilter("RPCUnpackingModule",
##     InputLabel = cms.InputTag("source"),
##     doSynchro = cms.bool(False)
## )
#process.load("EventFilter.RPCRawToDigi.RPCFrontierCabling_cfi")

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
process.rpcdigidqm.UseMuon =  cms.untracked.bool(useMuons)


########### RPC RecHit Probability #########
process.load("DQM.RPCMonitorDigi.RPCRecHitProbability_cfi")


################### FED ####################
process.load("DQM.RPCMonitorClient.RPCMonitorRaw_cfi")
process.load("DQM.RPCMonitorClient.RPCFEDIntegrity_cfi")
process.rpcFEDIntegrity.RPCRawCountsInputTag = 'provaDiNoCrash'
process.load("DQM.RPCMonitorClient.RPCMonitorLinkSynchro_cfi")


######### DQM Client Modules ###############
import DQM.RPCMonitorClient.RPCDqmClient_cfi

process.rpcdqmclientNOISE = DQM.RPCMonitorClient.RPCDqmClient_cfi.rpcdqmclient.clone(
    RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest", "RPCDeadChannelTest", "RPCClusterSizeTest", "RPCOccupancyTest","RPCNoisyStripTest"),
    DiagnosticPrescale = cms.untracked.int32(1),
    MinimumRPCEvents  = cms.untracked.int32(1),
    OfflineDQM = cms.untracked.bool(isOfflineDQM ),
    RecHitTypeFolder = cms.untracked.string("Noise")
    )

if useMuons :
    process.rpcdqmclientMUON = DQM.RPCMonitorClient.RPCDqmClient_cfi.rpcdqmclient.clone(
        RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest", "RPCDeadChannelTest", "RPCClusterSizeTest", "RPCOccupancyTest","RPCNoisyStripTest"),
        DiagnosticPrescale = cms.untracked.int32(1),
        MinimumRPCEvents  = cms.untracked.int32(1),
        OfflineDQM = cms.untracked.bool(isOfflineDQM),
        RecHitTypeFolder = cms.untracked.string("Muon")
        )

##### RPC RecHit Probability Client #######
process.load("DQM.RPCMonitorClient.RPCRecHitProbabilityClient_cfi")



############## Quality Tests ##############
process.qTesterRPC = cms.EDAnalyzer("QualityTester",
                                    qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
                                    prescaleFactor = cms.untracked.int32(1),
                                    qtestOnEndLumi = cms.untracked.bool(True),
                                    qtestOnEndRun = cms.untracked.bool(True)
                                    )

############## Chamber Quality ############
import DQM.RPCMonitorClient.RPCChamberQuality_cfi

process.rpcChamberQualityNOISE = DQM.RPCMonitorClient.RPCChamberQuality_cfi.rpcChamberQuality.clone(
    OfflineDQM = cms.untracked.bool(isOfflineDQM ),
    RecHitTypeFolder = cms.untracked.string("Noise"),
    MinimumRPCEvents  = cms.untracked.int32(1)
)

if useMuons :
    process.rpcChamberQualityMUON = DQM.RPCMonitorClient.RPCChamberQuality_cfi.rpcChamberQuality.clone(
        OfflineDQM = cms.untracked.bool(isOfflineDQM ),
        RecHitTypeFolder = cms.untracked.string("Muon"),
        MinimumRPCEvents  = cms.untracked.int32(1)
        )


################ Event Summary ###########
import DQM.RPCMonitorClient.RPCEventSummary_cfi

process.rpcEventSummaryNOISE = DQM.RPCMonitorClient.RPCEventSummary_cfi.rpcEventSummary.clone(
    OfflineDQM = cms.untracked.bool(isOfflineDQM ),
    MinimumRPCEvents  = cms.untracked.int32(1),
    RecHitTypeFolder = cms.untracked.string("Noise")
)

## process.rpcEventSummaryMUON = DQM.RPCMonitorClient.RPCEventSummary_cfi.rpcEventSummary.clone(
##     OfflineDQM = cms.untracked.bool(True),
##     MinimumRPCEvents  = cms.untracked.int32(10),
##     RecHitTypeFolder = cms.untracked.string("Muon")
## )


########### Message Logger ##############
process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('*'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('cout')
                                    )


############ Output Module ##############
process.out = cms.OutputModule("PoolOutputModule",
   fileName = cms.untracked.string('/tmp/cimmino/RPCDQM.root'),
   outputCommands = cms.untracked.vstring("keep *")
)


## ############ Memory check ##################
## process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
##         ignoreTotal = cms.untracked.int32(1) ## default is one
## ) 

## ################## Timing #################
## process.Timing = cms.Service("Timing")
## process.options = cms.untracked.PSet(
##     wantSummary = cms.untracked.bool(True)
##     )

## process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))


#########   Private Saver ###############
process.savedqmfile = cms.EDAnalyzer("SaveDQMFile",
                                     OutputFile = cms.untracked.string("/tmp/cimmino/DQM_165993.root")
                                     )


############# Path ######################

## process.rpcdqmsource = cms.Sequence(process.rpcdigidqm)
## process.rpcdqmclient = cms.Sequence(process.qTesterRPC * process.rpcdqmclient * process.rpcChamberQuality  * process.rpcEventSummary * process.dqmSaver)
## process.p = cms.Path(process.rpcdqmsource*process.rpcdqmclient)

process.rpcSourceSeq = cms.Sequence(process.rpcdigidqm*process.rpcrechitprobability*process.dqmEnv )

process.rpcClientNoiseSeq = cms.Sequence(process.rpcdqmclientNOISE * process.rpcChamberQualityNOISE  * process.rpcEventSummaryNOISE)

if useMuons:
    process.rpcClientMuonSeq = cms.Sequence(process.rpcdqmclientMUON * process.rpcChamberQualityMUON)

if useMuons:
    process.p = cms.Path(process.rpcSourceSeq  * process.qTesterRPC * process.rpcrechitprobabilityclient * process.rpcClientNoiseSeq  * process.rpcClientMuonSeq * process.savedqmfile)
else :
    process.p = cms.Path(process.rpcSourceSeq  * process.qTesterRPC * process.rpcrechitprobabilityclient * process.rpcClientNoiseSeq  * process.savedqmfile)    
#process.e = cms.EndPath(process.out)
