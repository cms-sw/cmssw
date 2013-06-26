import FWCore.ParameterSet.Config as cms

process = cms.Process("DataScoutingTest")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')




process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.DataScoutingTest = cms.EDAnalyzer("ScoutingTestAnalyzer",
  modulePath=cms.untracked.string("Test"),
  pfJetsCollectionName=cms.untracked.InputTag("hltCaloJetIDPassed")
  )

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('DataScoutingTest_DQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    )
)

process.dqmSaver.workflow = '/DataScouting/CMG/Test'
process.dqmsave_step = cms.Path(process.dqmSaver)


# Other statements
process.GlobalTag.globaltag = 'GR_R_52_V7::All'

# Path and EndPath definitions
process.testmodule_step = cms.Path(process.DataScoutingTest)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.schedule = cms.Schedule(process.testmodule_step,process.dqmsave_step,process.DQMoutput_step)

process.PoolSource.fileNames = ['/store/data/Run2012B/DataScouting/RAW/v1/000/194/535/16D13DD4-CBA2-E111-AE6F-001D09F24353.root']



