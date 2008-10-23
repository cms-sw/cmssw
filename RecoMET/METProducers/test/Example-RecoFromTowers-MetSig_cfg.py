import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration.StandardSequences.Digi_cff")

process.load("RecoMET.METProducers.CaloTowersOpt_cfi")

process.load("RecoLocalCalo.Configuration.RecoLocalCalo_cff")

process.load("RecoMET.Configuration.RecoMET_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_V9::All'


process.DQMStore = cms.Service("DQMStore")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_10/RelValQCD_Pt_120_170/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/7C9F0165-E59A-DD11-BE8D-001A92810AD2.root',
'/store/relval/CMSSW_2_1_10/RelValQCD_Pt_120_170/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/4EEE7ABD-F29A-DD11-9D4E-003048678DD6.root',
'/store/relval/CMSSW_2_1_10/RelValQCD_Pt_120_170/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/AC7C597E-F39A-DD11-85D4-001A92810AD0.root')
)

process.p = cms.Path(process.calotoweroptmaker*process.met*process.metsignificance)

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("./CaloMETSignif.root"),
                               outputCommands = cms.untracked.vstring("drop *",
                                                                      "keep *_*_*_TEST"
                                                                      )
                               )

process.e = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.p)
process.schedule.append(process.e)
