import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.TimerService = cms.Service("TimerService")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")

#
#  DQM SOURCES
#
process.load("DQMServices.Components.DQMEnvironment_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(9000)
)
process.source = cms.Source("PoolSource",

    #skipEvents = cms.untracked.uint32(3564),

    fileNames = 
cms.untracked.vstring(

       '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/CC80B73A-CA57-DE11-BC2F-000423D99896.root',
       '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/C68B7F1A-CD57-DE11-B706-00304879FA4A.root',
       '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/9CA9BBC1-CD57-DE11-B62D-001D09F2424A.root',
       '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/88AD5382-C657-DE11-831F-001D09F24498.root',
       '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/7C7CDD0F-C457-DE11-8EEE-000423D951D4.root',
       '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/4C30BDFF-B657-DE11-907A-001D09F24600.root',
       '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/383036B6-0458-DE11-819F-001D09F29524.root'

)

)

process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('*'),
    #debugModules = cms.untracked.vstring('hltResults'),
    #debugModules = cms.untracked.vstring('none-blah'),
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

process.psource = cms.Path(process.hltriggerResults)
process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


