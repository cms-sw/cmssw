import FWCore.ParameterSet.Config as cms

myWorkflow = '/My/Test/Workflow'
myWorkflowHarvesting = '/My/Test/WorkflowEDM_Harvesting'

process = cms.Process('HARVESTING')

# import of standard configurations
process.load("DQMServices.Core.DQM_cfg")
#process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('file:%s_MEtoEDM.root' % (myWorkflow[1:].replace('/', '__'))),
    processingMode = cms.untracked.string('RunsAndLumis')
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:com10', '')

# Path and EndPath definitions
process.load("DQMServices.Components.DQMFileSaver_cfi")
process.dqmSaver.saveByRun = cms.untracked.int32(1)
process.dqmSaver.workflow = cms.untracked.string(myWorkflowHarvesting)
process.edmtome_step = cms.Path(process.EDMtoME)
process.dqmsave_step = cms.Path(process.dqmSaver)

# Schedule definition
process.schedule = cms.Schedule(process.edmtome_step,process.dqmsave_step)
