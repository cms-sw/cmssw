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
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
)
process.source.fileNames = ['file:/media/usbdisk1/ZeeRelVal_311_FC71916C-756B-DE11-8631-000423D94700.root']


    

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




process.psource = cms.Path(process.egHLTOffDQMSource*process.egHLTOffDQMClient)
process.p = cms.EndPath(process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
#process.GlobalTag.globaltag = 'STARTUP::All'
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


