import FWCore.ParameterSet.Config as cms

process = cms.Process("VisDT")

# the source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/CRUZET3/Cosmics/RAW/v1/000/050/781/986D05EF-D44C-DD11-8F6C-000423D985E4.root',
    '/store/data/CRUZET3/Cosmics/RAW/v1/000/050/781/B284117B-CA4C-DD11-B34D-000423D9939C.root',
    '/store/data/CRUZET3/Cosmics/RAW/v1/000/050/781/C0D9C572-CA4C-DD11-AAE5-000423D98804.root'
    ))





process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )


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


