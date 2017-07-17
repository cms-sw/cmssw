import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Components.DQMEnvironment_cfi")


process.load("DQMOffline.Trigger.EgHLTOfflineSource_cfi")
process.load("DQMOffline.Trigger.EgHLTOfflineClient_cfi")


#load calo geometry
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")    
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input =cms.untracked.int32(10000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
)
process.source.fileNames = [
    
    #'/store/data/Run2011A/Photon/RECO/PromptReco-v4/000/168/437/3ACAB8FD-11A6-E011-894A-E0CB4E553651.root',
    #------------Run 172163------------
    #'/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/0A7A094C-0FBC-E011-B3BE-BCAEC518FF67.root',
    #'/store/data/Run2011A/SingleElectron/RECO/PromptReco-v5/000/172/163/0C163946-FCBB-E011-8C0C-BCAEC5329719.root',
  
    #-----run165364----------------
    #'/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/165/364/22FE09DD-CD84-E011-B073-001D09F24259.root',
    #'/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/165/364/28FF1200-B984-E011-A4A5-001D09F2B30B.root',

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
process.DQMStore.verbose = 1
#process.DQM.collectorHost = ''
#process.GlobalTag.globaltag = 'STARTUP::All'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/Run2011A/SingleElectronRuns165364-166462NoEtcut/RECO'
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = 1
#---Morse-----
process.dqmSaver.dirName = '/data/ndpc3/c/dmorse/HLTDQMrootFiles'
#-------


#process.p = cms.Sequence(process.dqmSaver)
process.psource = cms.Path(process.egHLTOffDQMSource*process.egHLTOffDQMClient)
process.p = cms.EndPath(process.dqmSaver)


