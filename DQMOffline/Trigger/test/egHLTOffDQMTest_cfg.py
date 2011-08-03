import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Components.DQMEnvironment_cfi")


process.load("DQMOffline.Trigger.EgHLTOfflineSource_cfi")
process.load("DQMOffline.Trigger.EgHLTOfflineClient_cfi")


#load calo geometry
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")    
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input =cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
)
process.source.fileNames = [
    #------------Run 172163-------
 #   '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/0A7A094C-0FBC-E011-B3BE-BCAEC518FF67.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/0C163946-FCBB-E011-8C0C-BCAEC5329719.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/0E1300F8-0BBC-E011-B641-BCAEC518FF6E.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/2A95A64B-0FBC-E011-AD06-BCAEC53296FC.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/3EA34830-20BC-E011-8F3D-003048F024F6.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/708D414A-0FBC-E011-8E25-BCAEC5364C42.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/72C4E527-04BC-E011-A7D6-BCAEC53296FB.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/8022C68F-13BC-E011-A28B-BCAEC53296FC.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/926D9A90-13BC-E011-9A66-485B39897227.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/AC5DCC27-04BC-E011-9021-E0CB4E4408D1.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/B0EB7091-13BC-E011-9E4E-003048D375AA.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/C2621DF5-0BBC-E011-8229-BCAEC518FF50.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/EA6B5B29-04BC-E011-A68C-BCAEC5329708.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/EC76DB4A-0FBC-E011-B728-BCAEC5329728.root',
  #  '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/FC5DC7D1-FFBB-E011-94CD-BCAEC5364C62.root',
      

    ]

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    debugInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('EgammaHLTOffline','EgHLTOfflineClient'),
    destinations = cms.untracked.vstring('debugInfo', 
        'detailedInfo', 
        'critical', 
        'cout')
 )

process.DQMStore.collateHistograms = True
#process.DQMStore.verbose = 0
#process.DQM.collectorHost = ''
#process.GlobalTag.globaltag = 'STARTUP::All'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/Run2011A/SingleElectronRuns165364-166462NoEtcut/RECOTRASH'
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = 1
#---Morse-----
process.dqmSaver.dirName = '/data/ndpc3/c/dmorse/HLTDQMrootFiles'
#-------


#process.p = cms.Sequence(process.dqmSaver)
process.psource = cms.Path(process.egHLTOffDQMSource*process.egHLTOffDQMClient)
process.p = cms.EndPath(process.dqmSaver)


