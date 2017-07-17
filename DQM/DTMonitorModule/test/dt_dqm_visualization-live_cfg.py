import FWCore.ParameterSet.Config as cms

process = cms.Process("VisDT")

process.load("DQM.Integration.test.inputsource_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("DQM.DTMonitorModule.dt_dqm_visualization_common_cff")



# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

process.dtDQMPath = cms.Path(process.calibrationEventsFilter * process.reco)




