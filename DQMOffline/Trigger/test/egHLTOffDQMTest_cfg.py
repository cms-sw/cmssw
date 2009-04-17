import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Components.DQMEnvironment_cfi")


process.load("DQMOffline.Trigger.EgammaHLTOffline_cfi")
process.load("DQMOffline.Trigger.EgHLTOffClient_cfi")
#process.load("DQMOffline.Trigger.relVal_2_1_0_Zee_small_cff");
process.load("DQMOffline.Trigger.relVal_Zee_219_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#process.PoolSource.fileNames = ['file:/scratch/sharper/cmsswDataFiles/zee_relVal_219.root']


#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring ('dummy') )
    

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




process.psource = cms.Path(process.egammaHLTDQM*process.egHLTOffDQMClient)
process.p = cms.EndPath(process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
#process.GlobalTag.globaltag = 'STARTUP::All'
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
process.PoolSource.fileNames = ['file:/scratch/sharper/RelValZEE_218_1.root']
#process.PoolSource.fileNames = ['file:/media/usbdisk2/RelValZEE_CMSSW_2_1_10/0A990453-9699-DD11-B9FD-001617C3B69C.root']
#process.PoolSource.fileNames= ['file:/media/usbdisk2/RelValZEE_CMSSW_2_1_7/0C3B40D7-F87D-DD11-A9FB-000423D998BA.root']
#process.PoolSource.fileNames= ['file:/media/usbdisk2/RelValZEE_CMSSW_2_1_7/0C3B40D7-F87D-DD11-A9FB-000423D998BA.root','file:/media/usbdisk2/RelValZEE_CMSSW_2_1_7/3A5455F3-F87D-DD11-AEF4-000423D94534.root']

