import FWCore.ParameterSet.Config as cms

process = cms.Process("VisDT")

# the source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Commissioning09/Cosmics/RAW/v1/000/084/189/46C061A8-2E3B-DE11-8608-001D09F24FEC.root'
    ))





process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("DQM.DTMonitorModule.dt_dqm_visualization_common_offline_cff")
#process.load("Configuration.StandardSequences.MagneticField_cff")



# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

process.dtVisPath = cms.Path(process.calibrationEventsFilter * process.reco)


# f = file('aNewconfigurationFile.cfg', 'w')
# f.write(process.dumpConfig())
# f.close()


