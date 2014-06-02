import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
)

process.pb_writer = cms.EDAnalyzer("DQMStoreWriter")

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step3 nevts:100'),
    name = cms.untracked.string('Applications')
)

#DQMFileSaver configuration
process.dqmSaver.saveByLumiSection = cms.untracked.int32(1)
process.dqmSaver.convention = cms.untracked.string('FilterUnit')
process.dqmSaver.fileFormat = cms.untracked.string('PB')
process.dqmSaver.workflow = cms.untracked.string('')

# Path and EndPath definitions
process.writer = cms.Path(process.pb_writer)
process.DQMoutput_step = cms.EndPath(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(process.writer, process.DQMoutput_step)
