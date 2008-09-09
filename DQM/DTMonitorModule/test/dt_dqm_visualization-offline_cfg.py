import FWCore.ParameterSet.Config as cms

process = cms.Process("VisDT")

# the source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Commissioning08/BeamHalo/RECO/StuffAlmostToP5_v1/000/061/642/10A0FE34-A67D-DD11-AD05-000423D94E1C.root'
    ))





process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )


process.load("DQM.Integration.test.dt_dqm_visualization_common_offline_cff")



# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

process.dtDQMPath = cms.Path(process.calibrationEventsFilter * process.reco)


# f = file('aNewconfigurationFile.cfg', 'w')
# f.write(process.dumpConfig())
# f.close()


