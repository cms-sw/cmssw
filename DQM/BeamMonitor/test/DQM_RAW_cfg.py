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
process.GlobalTag.globaltag = 'MC_31X_V9::All'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3200)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/B49925AC-E3BC-DE11-B2B4-00261894395C.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/9E031261-E4BC-DE11-B247-001A92971B68.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/8E5D9592-4EBD-DE11-880E-001A92971B0E.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/7E33DF20-DEBC-DE11-ABF9-002618943875.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/78493C41-E2BC-DE11-86CD-0018F3D096E6.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/6E42191F-DEBC-DE11-B187-00248C0BE013.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/5C2F915E-DDBC-DE11-BD89-001BFCDBD15E.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/463F7C19-E3BC-DE11-A8CC-002618943833.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/2AB8BCFB-E5BC-DE11-B550-00261894389A.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/20D9811D-DEBC-DE11-B67E-002618943829.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/00864FB5-E1BC-DE11-8613-0018F3D0961A.root',
       '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/0012B75C-DDBC-DE11-81E2-002618943921.root'
    )
)

process.RecoForDQM = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks)

process.pp = cms.Path(process.RecoForDQM*process.dqmBeamMonitor+process.dqmBeamCondMonitor+process.dqmEnv+process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = 'cmslpc08.fnal.gov'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )


