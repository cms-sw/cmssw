import FWCore.ParameterSet.Config as cms

process = cms.Process("VisDT")

# the source
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring('file:/data1/lookarea/GlobalAug07.00017220.0001.A.storageManager.0.0000.dat')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("DQM.Integration.test.dt_dqm_visualization_common_cff")



# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

process.dtDQMPath = cms.Path(process.calibrationEventsFilter * process.reco)


# f = file('aNewconfigurationFile.cfg', 'w')
# f.write(process.dumpConfig())
# f.close()


