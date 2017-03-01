##-- Starting
import FWCore.ParameterSet.Config as cms
process = cms.Process("DQM")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.EventContent.EventContent_cff')


##-- DQM Loading
# DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
# DQM Sources
process.load("CondCore.DBCommon.CondDBSetup_cfi")
# DQMOffline/Trigger
#process.load("DQMOffline.Trigger.JetMETHLTOfflineSource_cfi")
process.load("DQMOffline.Trigger.JetMETHLTOfflineAnalyzer_cff")
process.load("DQMOffline.Trigger.JetMETHLTOfflineClient_cfi")
process.load("DQMOffline.Trigger.HLTJetMETQualityTester_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
#
process.DQMStore.verbose = 0 #0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online' #Online
process.dqmSaver.saveByRun = 1 #0
process.dqmSaver.saveAtJobEnd = True


##-- GlobalTag
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'GR_R_71_V1::All'

#######
# Other statements
#from Configuration.Applications.ConfigBuilder import ConfigBuilder
#process.DQMOffline.visit(ConfigBuilder.MassSearchReplaceProcessNameVisitor("HLT", "reHLT", whitelist = ("subSystemFolder",)))
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:com10_GRun', '')


##-- DQMOffline/Trigger
from DQMOffline.Trigger.JetMETHLTOfflineClient_cfi import *
process.load("DQMServices.Components.DQMStoreStats_cfi")


##-- Source
# Note: We need RECO here (not AOD), because of JetHelper Class
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

# Input source
process.source = cms.Source("PoolSource",
        secondaryFileNames = cms.untracked.vstring(),
        fileNames = cms.untracked.vstring('/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/008429E1-53A1-E311-81A7-02163E00A313.root',
                                          '/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/08367C14-3CA1-E311-A4BE-0025904B2C74.root',
                                          '/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/08983374-3BA1-E311-B151-02163E009E00.root',
                                          '/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/0A1C5BCE-51A1-E311-B40C-02163E008F35.root',
                                          '/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/12741B3D-40A1-E311-BB08-0025904B26B4.root',
                                          '/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/1E090904-4AA1-E311-BF2A-02163E00E9E8.root',
                                          '/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/1EBC3C10-4DA1-E311-806C-02163E00E7AC.root',
                                          '/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/201C9342-35A1-E311-8CC8-02163E00E945.root',
                                          '/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/20F4610D-69A1-E311-8EBC-0025904B26B2.root',
                                          '/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/20FD35C2-3BA1-E311-A72F-02163E00E72D.root',
                                          '/store/relval/CMSSW_7_1_0_pre3/JetHT/RECO/GR_R_71_V1_RelVal_jet2012C-v1/00000/FE32B28F-60A1-E311-B9A1-02163E00E5CA.root'),
                            lumisToProcess = cms.untracked.VLuminosityBlockRange("199812:70-199812:141", "199812:144-199812:163", "199812:182-199812:211", "199812:214-199812:471", "199812:474-199812:505", "199812:508-199812:557", "199812:560-199812:571", "199812:574-199812:623", "199812:626-199812:751", "199812:754-199812:796")
                            )




##-- Output
process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands, #DQMEventContent
    fileName = cms.untracked.string('JetMET_DQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    )
)

##-- Logger
process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'cout')
)


##-- Config
from DQMOffline.Trigger.JetMETHLTOfflineSource_cfi import *
process.jetMETHLTOfflineSource.processname = cms.string("reHLT")
process.jetMETHLTOfflineSource.triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","reHLT")
process.jetMETHLTOfflineSource.triggerResultsLabel = cms.InputTag("TriggerResults","","reHLT")
process.jetMETHLTOfflineSource.plotEff = cms.untracked.bool(True)


##-- Let's it runs
process.JetMETSource_step = cms.Path( process.jetMETHLTOfflineAnalyzer )
process.JetMETClient_step = cms.Path( process.jetMETHLTOfflineClient )
process.dqmsave_step      = cms.Path( process.dqmSaver )
process.DQMoutput_step    = cms.EndPath( process.DQMoutput )
# Schedule
process.schedule = cms.Schedule(process.JetMETSource_step,
                                process.JetMETClient_step,
                                process.dqmsave_step,
                                process.DQMoutput_step)
