import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#       '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0006/34733721-5278-DE11-BA19-001D09F2A49C.root',
#       '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0006/9EB9610F-5278-DE11-9F3A-001D09F26509.root',
#       '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0006/AA2E6BD1-5078-DE11-A537-0019B9F705A3.root',
#       '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0006/B2441B1B-5278-DE11-93F6-000423D99E46.root',
#       '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0006/C28BCD25-5278-DE11-9EC6-001D09F24EAC.root'

	'/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/1A471008-83B4-DE11-8CB8-000423D9890C.root',
	'/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/4479D311-87B4-DE11-A495-000423D9890C.root',
	'/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/8A23E975-BDB4-DE11-A0ED-0019B9F6C674.root',
	'/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/8A2431FE-87B4-DE11-BEE6-000423D98800.root',
	'/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/961D5F56-89B4-DE11-8C30-000423D98EC4.root'

    )
    , duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

)

process.pp = cms.Path(process.dqmBeamMonitor+process.dqmBeamCondMonitor+process.dqmEnv+process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = 'cmslpc08.fnal.gov'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(cms.PSet(
    record = cms.string('BeamSpotObjectsRcd'),
    tag = cms.string('Early10TeVCollision_3p8cm_v3_mc_IDEAL')
    )),
#    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT')
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_BEAMSPOT')
)

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.schedule = cms.Schedule(process.pp)

