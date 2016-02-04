import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQM.HLTEvF.HLTMonitor_cff")

process.load("DQM.HLTEvF.HLTEventInfoClient_cff")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:HLTFromDigiRaw.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
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

process.dumpcont = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.EndPath(process.dqmEnv*process.dqmSaver)
process.PoolSource.fileNames = ['file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 
    'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root', 'file:/tmp/bjbloom/0491EAAC-DF19-DD11-AECD-000423D98950.root']
process.DQMStore.verbose = 0
process.DQM.collectorHost = 'srv-c2d05-12'
process.DQM.collectorPort = 9190
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'HLT'
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = False

