import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

# my client and my Tests
process.load('DQMServices.Examples.test.DQMExample_Step2_cfi')
process.load('DQMServices.Examples.test.DQMExample_GenericClient_cfi')
process.load('DQMServices.Examples.test.DQMExample_qTester_cfi')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:OUT_step1.root"))


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')  #for MC



# Path and EndPath definitions
process.myHarvesting = cms.Path(process.DQMExampleStep2)
process.myEff = cms.Path(process.DQMExampleGenericClient)
process.myTest = cms.Path(process.DQMExample_qTester)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(
                                process.myEff,
                                process.myTest,
                                process.myHarvesting,
                                process.dqmsave_step
    )

process.DQMStore.verbose =  cms.untracked.int32(1)
process.DQMStore.verboseQT =  cms.untracked.int32(1)

#process.DQMStore.collateHistograms = cms.untracked.bool(True)
#process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
#process.dqmSaver.forceRunNumber = cms.untracked.int32(123456)

process.dqmSaver.workflow = '/TTbarLepton/myTest/DQM'
