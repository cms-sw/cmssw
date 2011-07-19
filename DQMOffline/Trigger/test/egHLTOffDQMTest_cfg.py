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
    input =cms.untracked.int32(5000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
)
process.source.fileNames = [
   '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/166/033/00718196-658E-E011-A760-003048D2BE12.root',
    '/store/data/Run2011A/SingleElectron/RECO/PromptReco-v4/000/165/088/98CB07EF-E37F-E011-BAC2-003048F1183E.root',
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
process.dqmSaver.workflow = '/Run2011A/SingleElectronJune10/RECO'
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = 1
#---Morse-----
process.dqmSaver.dirName = '/data/ndpc3/c/dmorse/HLTDQMrootFiles/4_2Patch'
#-------

process.psource = cms.Path(process.egHLTOffDQMSource*process.egHLTOffDQMClient)
process.p = cms.EndPath(process.dqmSaver)


