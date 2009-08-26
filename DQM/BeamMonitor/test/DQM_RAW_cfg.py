import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.BeamMonitor.BeamMonitor_MC_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

### conditions
process.GlobalTag.globaltag = 'MC_31X_V3::All'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3200)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/EC49FE0B-DB90-DE11-8434-001D09F2545B.root',
        '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/C0F33D87-D990-DE11-8490-000423D6B444.root',
        '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/B69156C8-D790-DE11-B825-0030487C6090.root',
        '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/8273D625-D990-DE11-ADA8-0019B9F709A4.root',
        '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/74268A17-DA90-DE11-8DB4-001D09F2545B.root',
        '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/5C52DCBD-D790-DE11-BB37-000423D992A4.root',
        '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/503DBE8C-D990-DE11-9BDC-000423D9939C.root',
        '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/46D0F7EC-D990-DE11-9EC2-001D09F231B0.root',
        '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/400CCE1C-DA90-DE11-A2B0-001D09F231B0.root'
    )
)

process.raw2digi_step = cms.Sequence(process.siPixelDigis*
                                     process.SiStripRawToDigis
                                     )
process.tracking_step = cms.Sequence(process.trackerlocalreco+
                                     process.offlineBeamSpot+
                                     process.recopixelvertexing*
                                     process.ckftracks
                                     )

process.pt = cms.Path(process.raw2digi_step+process.tracking_step)
process.pp = cms.Path(process.dqmBeamMonitor+process.dqmBeamCondMonitor+process.dqmEnv+process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = 'cmslpc02.fnal.gov'
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

#process.schedule = cms.Schedule(process.pt,process.pp,process.end)
process.schedule = cms.Schedule(process.pt,process.pp)

