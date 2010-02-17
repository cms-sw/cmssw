import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMTest")

#set DQM enviroment
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#load and setup E/g HLT Offline DQM module
process.load("DQMOffline.Trigger.EgHLTOfflineSource_cfi")
process.load("DQMOffline.Trigger.TopElectronHLTOfflineSource_cfi")
#load calo geometry
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
# Other statements
process.GlobalTag.globaltag = 'START3X_V21::All'

#configure message logger to something sane
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cout.threshold = cms.untracked.string('WARNING')
process.MessageLogger.cerr.FwkSummary = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(500),
    limit = cms.untracked.int32(10000000)
)
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(500),
    limit = cms.untracked.int32(10000000)
)


process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
)
process.source.fileNames=[
       '/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0014/A43F691A-6D13-DF11-99D0-001A92971BDC.root',
       '/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0013/F0BF21B7-5513-DF11-8A40-0018F3D09688.root',
       '/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0013/CA5E6D51-5113-DF11-8CC2-001A92811748.root',
       '/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0013/82ABD700-5613-DF11-92D2-0018F3D09664.root',
       '/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0013/3C397451-6213-DF11-B274-0018F3D095FA.root',
       '/store/relval/CMSSW_3_5_0/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0013/385227FB-5413-DF11-8E1A-0018F3D09634.root' ]


process.maxEvents = cms.untracked.PSet(
  
    input = cms.untracked.int32(-1)
)


process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")    
process.load("Configuration.EventContent.EventContent_cff")


process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RECO')),
    fileName = cms.untracked.string('test/relVal_350pre2Test.root')
)
process.FEVT.outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_DQMTest')


#monitor elements are converted to EDM format to store in CMSSW file
#client will convert them back before processing
process.psource = cms.Path(process.topElectronHLTOffDQMSource*process.egHLTOffDQMSource*process.hltTrigReport*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''

