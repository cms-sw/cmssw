import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3200)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0006/34733721-5278-DE11-BA19-001D09F2A49C.root',
       '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0006/9EB9610F-5278-DE11-9F3A-001D09F26509.root',
       '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0006/AA2E6BD1-5078-DE11-A537-0019B9F705A3.root',
       '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0006/B2441B1B-5278-DE11-93F6-000423D99E46.root',
       '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0006/C28BCD25-5278-DE11-9EC6-001D09F24EAC.root'
    )
)

process.pp = cms.Path(process.dqmBeamMonitor+process.dqmBeamCondMonitor+process.dqmEnv+process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = 'cmslpc03.fnal.gov'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
#process.hltResults.plotAll = True
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('test.root'),
			       outputCommands = cms.untracked.vstring(
				'drop *',
				'keep *_offlineBeamSpot_*_*',
				'keep *_generalTracks_*_*'
			       )
                              )

process.end = cms.EndPath(process.out)

#process.schedule = cms.Schedule(process.pp,process.end)
process.schedule = cms.Schedule(process.pp)

