import FWCore.ParameterSet.Config as cms

nevts=200
histofile="histo.root"

process = cms.Process("Demo")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(nevts)
)

# use TFileService for output histograms
process.TFileService = cms.Service("TFileService",
                              fileName = cms.string(histofile)
                              )

process.source = cms.Source("PoolSource",
                           fileNames = cms.untracked.vstring(
        # TTbar RelVal 2.1.0
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/02F674DE-A160-DD11-A882-001617DBD5AC.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/04327AC0-1C61-DD11-93B8-001BFCDBD19E.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/06621B92-A060-DD11-B33C-000423D6CA6E.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/08059389-0E61-DD11-89D1-001A928116DC.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/0830A57C-1561-DD11-9B9D-001731A28A31.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/0C6AEA0F-0E61-DD11-BF9F-0018F3D096E0.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/0C74136A-1761-DD11-80D7-0018F3D09686.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/18D5104F-A060-DD11-8746-000423D991D4.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/1ED2C659-1861-DD11-857C-0017312B5F3F.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/1EDB2A98-0D61-DD11-9D88-003048767DDB.root'
           )
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.SingleJetAnalyser = cms.EDFilter("JetMETHLTAnalyzer",
                                 Debug    = cms.bool(True),
                                 Progress  = cms.int32(1),
                                 CaloJets = cms.string('ak4CaloJets'),
                                 GenJets  = cms.string('ak4GenJets'),
                                 HLTriggerResults = cms.InputTag("TriggerResults::HLT"),
                                 l1extramc  = cms.string('l1extraParticles')
                                 )

process.p1 = cms.Path(process.SingleJetAnalyser)

process.schedule = cms.Schedule(process.p1)
