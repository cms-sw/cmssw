# The following comments couldn't be translated into the new config version:

#
#  keep only muon-related info here
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("MISO")
process.load("Configuration.EventContent.EventContent_cff")

#    service = MessageLogger {
#       untracked vstring destinations = { "cout" }
#       untracked vstring debugModules = { "muIsoDepositTk", 
#              "muIsoDepositCalByAssociatorHits", 
#              "muIsoDepositCalByAssociatorTowers", 
#              "muIsoDepositCal" }
#       untracked vstring categories = { "RecoMuon" , "MuonIsolation" }
#
#       untracked PSet cout = { 
#         untracked string threshold = "DEBUG" 
#         untracked int32 lineLength  = 132
#         untracked bool noLineBreaks = true
#         untracked PSet DEBUG = {untracked int32 limit = 0 }
#         untracked PSet RecoMuon = { untracked int32 limit = 10000000}
#         untracked PSet MuonIsolation = { untracked int32 limit = 10000000}
#       }
#    }
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")

#process.load("RecoMuon.Configuration.RecoMuon_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

#process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

#has everything(?) one needs
# pick muIsolation sequence for "standard" iso reco for tracker and global muons
process.load("RecoMuon.MuonIsolationProducers.muIsolation_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/2007/12/7/RelVal-RelValBJets_Pt_50_120-1197045102/0002/0A21A5F4-02A5-DC11-89F5-000423DD2F34.root')
)
process.source = cms.Source ("PoolSource",
       fileNames = cms.untracked.vstring (
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/10438122-2A5F-DD11-A77F-000423D985E4.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/12F34420-2A5F-DD11-AB6E-000423D6CA6E.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/244E7C0B-315F-DD11-ACFC-001617E30F58.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/2ADD8A12-315F-DD11-8AB8-000423D6C8E6.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/34A291FB-305F-DD11-833E-001617C3B6CC.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/383E09CA-2C5F-DD11-9A28-000423D6BA18.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/40F0F8A4-2A5F-DD11-BC72-001617C3B64C.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/4AD39C8C-2A5F-DD11-B935-001617C3B710.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/4C0D4911-315F-DD11-A20D-001617DBD332.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/4C32E425-2A5F-DD11-B819-000423D6C8EE.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/50881CBB-2A5F-DD11-92C6-001617C3B6E8.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/52B83F75-2A5F-DD11-AD56-001617C3B6CC.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/544DC99A-2A5F-DD11-9160-001617C3B6E2.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/62F7698D-2A5F-DD11-907A-001617C3B6DC.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/7C8A2791-2A5F-DD11-814D-001617DBCF6A.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/7EDA5005-315F-DD11-8019-001617C3B706.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/8A91E518-2A5F-DD11-B49A-000423D6B42C.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/8CC497AE-2A5F-DD11-AE43-000423DD2F34.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/9A469FA8-2A5F-DD11-9909-001617C3B6FE.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/9A5BE3A4-2A5F-DD11-A61B-001617DF785A.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/9AC2141C-2A5F-DD11-ADF5-000423D6A6F4.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/9CCFA319-2A5F-DD11-B0AA-000423D94700.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/A0F6C41D-2A5F-DD11-8685-000423D6BA18.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/B0159DAC-2A5F-DD11-98A8-001617E30D00.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/B05C32FC-305F-DD11-A957-001617C3B70E.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/C6ADD999-2A5F-DD11-AF9F-0016177CA7A0.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/C8AEE585-2A5F-DD11-BB37-001617C3B77C.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/CC5178C4-2A5F-DD11-BCE6-001617E30F4C.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/CE9FE020-2A5F-DD11-9846-000423D6CA72.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/D24BFA7E-2A5F-DD11-8F79-001617C3B70E.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/D62761FA-305F-DD11-A108-0016177CA778.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/DA0DDFB6-2A5F-DD11-987A-001617DBD5B2.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/E64386FE-305F-DD11-BA68-0019DB29C614.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/E6BC0D37-2A5F-DD11-9ACB-000423D6B444.root',
       '/store/relval/CMSSW_2_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v2/0000/F251D794-2A5F-DD11-BA5D-00161757BF42.root'
      ),
       secondaryFileNames = cms.untracked.vstring (
      )
)

process.RECO = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('file:isoTest.root')
)

process.p1 = cms.Path(process.muIsolation)
process.outpath = cms.EndPath(process.RECO)
process.RECO.outputCommands.append('drop *_*_*_*')
process.RECO.outputCommands.extend(process.RecoMuonRECO.outputCommands)

