import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Components.DQMEnvironment_cfi")


process.load("DQMOffline.Trigger.EgammaHLTOffline_cfi")
process.load("DQMOffline.Trigger.EgHLTOffClient_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/6/RelVal-RelValZMM-1212543891-STARTUP-2nd-02/0000/4A3DAF35-E533-DD11-9D80-001617E30F56.root')
)

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
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('debugInfo', 
        'detailedInfo', 
        'critical', 
        'cout')
)




process.psource = cms.Path(process.egammaHLTDQM*process.egHLTOffDQMClient)
process.p = cms.EndPath(process.dqmSaver)
process.PoolSource.fileNames = ['file:/scratch/sharper/DQMFiles/Zee_RelVal_210pre9_1.root']
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
#process.GlobalTag.globaltag = 'STARTUP::All'
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

