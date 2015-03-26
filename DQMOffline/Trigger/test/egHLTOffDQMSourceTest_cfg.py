import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMTest")

#set DQM enviroment
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#load and setup E/g HLT Offline DQM module
process.load("DQMOffline.Trigger.EgHLTOfflineSource_cfi")
#load calo geometry
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
# Other statements
#use two following lines to grab GlobalTag automatically
#from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond['hltonline']
process.GlobalTag.globaltag = 'GR_R_50_V11::All'
#configure message logger to something sane
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cout.threshold = cms.untracked.string('WARNING')
process.MessageLogger.cerr.FwkSummary = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(500),
    limit = cms.untracked.int32(10000000)
)
#process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(True))

#process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
#    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
#)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
)
process.source.fileNames=[
       #'/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0014/A43F691A-6D13-DF11-99D0-001A92971BDC.root',
       #'/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0013/F0BF21B7-5513-DF11-8A40-0018F3D09688.root',
       #'/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0013/CA5E6D51-5113-DF11-8CC2-001A92811748.root',
       #'/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0013/82ABD700-5613-DF11-92D2-0018F3D09664.root',
       #'/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0013/3C397451-6213-DF11-B274-0018F3D095FA.root',
       #'/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0013/385227FB-5413-DF11-8E1A-0018F3D09634.root'
    
     '/store/relval/CMSSW_5_2_0_pre3/SingleElectron/RECO/GR_R_50_V11_RelVal_electron2011B-v1/0000/0CCB804F-154C-E111-A8A7-001A92810A94.root',
     '/store/relval/CMSSW_5_2_0_pre3/SingleElectron/RECO/GR_R_50_V11_RelVal_electron2011B-v1/0000/740FF94D-154C-E111-95C8-001A928116C0.root',
     '/store/relval/CMSSW_5_2_0_pre3/SingleElectron/RECO/GR_R_50_V11_RelVal_electron2011B-v1/0000/92D7A150-154C-E111-94C9-003048FFD7C2.root',
     '/store/relval/CMSSW_5_2_0_pre3/SingleElectron/RECO/GR_R_50_V11_RelVal_electron2011B-v1/0000/96F9074C-154C-E111-BD78-001A92810AA6.root',
     '/store/relval/CMSSW_5_2_0_pre3/SingleElectron/RECO/GR_R_50_V11_RelVal_electron2011B-v1/0000/DEDE1B51-154C-E111-9CD6-001A92810AEE.root',
     '/store/relval/CMSSW_5_2_0_pre3/SingleElectron/RECO/GR_R_50_V11_RelVal_electron2011B-v1/0000/ECFBE94D-154C-E111-B5A8-001A92971BC8.root',
  #   '/store/relval/CMSSW_5_2_0_pre3/SingleElectron/RECO/GR_R_50_V11_RelVal_electron2011A-v1/0000/60F854F0-504C-E111-9C47-002618943932.root',
  #   '/store/relval/CMSSW_5_2_0_pre3/SingleElectron/RECO/GR_R_50_V11_RelVal_electron2011A-v1/0000/EAD4E6ED-504C-E111-9AF3-003048678A7E.root',

    #  '/store/data/Run2011A/DoubleElectron/RECO/PromptReco-v4/000/165/088/66AD1342-E47F-E011-B825-003048F01E88.root',
    #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/165/098/203C4130-DA7F-E011-9BD0-003048F11CF0.root',
    #  '/store/data/Run2011A/DoubleElectron/RECO/PromptReco-v4/000/165/098/C4A7FB10-DA7F-E011-97CF-0030487CD718.root',
    #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/165/099/3A9A0435-D17F-E011-9999-0030487C6A66.root',
    #  '/store/data/Run2011A/DoubleElectron/RECO/PromptReco-v4/000/165/099/82CDB84B-D17F-E011-8C79-0030487CD710.root',
    #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/165/102/2A1014DE-CF80-E011-9924-001617DBD5AC.root',
    #  '/store/data/Run2011A/DoubleElectron/RECO/PromptReco-v4/000/165/102/00530480-CF80-E011-B76C-001617E30D4A.root',
    #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/165/103/0EB41167-EE80-E011-8B64-003048F024DC.root',
    #  '/store/data/Run2011A/DoubleElectron/RECO/PromptReco-v4/000/165/103/B0BACE7E-EE80-E011-886C-00304879BAB2.root',
    #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/0C163946-FCBB-E011-8C0C-BCAEC5329719.root',
        #-----test across runs----
        #   '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/165/467/7E4DDAD6-ED85-E011-AC8D-001D09F24600.root',
        #   '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/165/472/361C4077-3386-E011-A483-001D09F2960F.root',

        ]

process.maxEvents = cms.untracked.PSet(

    input = cms.untracked.int32(-1)
)


process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")    
process.load("Configuration.EventContent.EventContent_cff")


process.DQMStore.collateHistograms = True

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RECO')),
 #   fileName = cms.untracked.string('/data/ndpc3/c/dmorse/HLTDQMrootFiles/May18/SourceTest_420_2.root')
 #   fileName = cms.untracked.string('Run2011A_SingleElectronRuns165364-166462Et40cut_RECO.root')
    fileName = cms.untracked.string('SingleElectron_CMSSW_5_2_0_pre3_RECO_2011B.root')
)
process.FEVT.outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_DQMTest')


#monitor elements are converted to EDM format to store in CMSSW file
#client will convert them back before processing
process.psource = cms.Path(process.egHLTOffDQMSource*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)
process.MEtoEDMConverter.Verbosity = 0
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''

