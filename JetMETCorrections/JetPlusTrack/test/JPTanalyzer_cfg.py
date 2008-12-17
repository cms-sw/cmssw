import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("RECO3")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

from JetMETCorrections.Configuration.JetPlusTrackCorrections_cff import *
#JetPlusTrackZSPCorrectorIcone5.NonEfficiencyFileResp = cms.string('CMSSW_167_TrackLeakage_one')
#JetPlusTrackZSPCorrectorIcone5.NonEfficiencyFile = cms.string('CMSSW_167_TrackNonEff_one')

process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")

# build gen jet without neutrinos
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.genParticlesForJets.ignoreParticleIDs = cms.vuint32(
   1000022, 2000012, 2000014,
   2000016, 1000039, 5000039,
   4000012, 9900012, 9900014,
   9900016, 39, 12, 14, 16
)
process.genParticlesForJets.excludeFromResonancePids = cms.vuint32(12, 14, 16)

from RecoJets.JetProducers.iterativeCone5GenJets_cff import iterativeCone5GenJets
process.iterativeCone5GenJetsNoNuBSM  =  iterativeCone5GenJets.clone()

# test QCD file from 210 RelVal is on /castor/cern.ch/user/a/anikiten/jpt210qcdfile/
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(30000)
)

process.source = cms.Source("PoolSource",
# /castor/cern.ch/user/p/pjanot/CMSSW219/
# reco_QCDptxxx_yyy_Full.root, xxx = 20, 30, 50, 80, 120, 160, 250, 350 and 500, and yyy = 30, 50, 80, 120,
# 160, 250, 350, 500 and 700,
# cmssw210
#    fileNames = cms.untracked.vstring('file:/tmp/anikiten/FC999068-DB60-DD11-9694-001A92971B16.root')
# cmssw218
#    fileNames = cms.untracked.vstring('file:/tmp/anikiten/0C66A939-8F82-DD11-8442-0019DB29C614.root')
#     fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_8/RelValBJets_Pt_50_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/0C66A939-8F82-DD11-8442-0019DB29C614.root')
# /RelValQCD_Pt_80_120/CMSSW_2_1_8_IDEAL_V9_v1/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO
     fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/000AD2A4-6E86-DD11-AA99-000423D9863C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/02D641CC-6D86-DD11-B1AA-001617C3B64C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/0876950D-6F86-DD11-B4B8-000423D6A6F4.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/0AC366B7-6D86-DD11-8695-001617E30D0A.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/1415D0A5-6E86-DD11-8F0C-001617C3B778.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/145FBEB9-6D86-DD11-BD98-001617DF785A.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/1815C2A0-6E86-DD11-A160-001617E30F58.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/1869C2C0-6D86-DD11-ADB6-001617C3B76E.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/186A3FC0-6D86-DD11-B020-000423D6CA72.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/1A688BC9-6D86-DD11-8C60-001617DBD5AC.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/20EBA2C9-6D86-DD11-88EC-001617E30D40.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/24822ABB-6D86-DD11-B76B-001617E30F4C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/24E0B0C7-6D86-DD11-8C09-0019DB29C5FC.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/2652503A-6F86-DD11-9552-001617E30F4C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/26E79ACA-6D86-DD11-8A10-001617E30D00.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/2A0341A9-6E86-DD11-9AE4-000423D6CA02.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/3038D377-6E86-DD11-A974-001617C3B5E4.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/369AEBC5-6D86-DD11-B911-001617E30D4A.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/3841A6BC-6D86-DD11-9D84-001617E30D52.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/3C049BBF-6D86-DD11-8D0D-001617DBD556.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/3EBAB9A6-6E86-DD11-AB4D-001617C3B70E.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/3EBCA837-6F86-DD11-8892-001617E30D4A.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/3EDB9AD6-6E86-DD11-9EE3-000423D992A4.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/42CED2B7-6D86-DD11-B551-001617C3B6E2.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/441FDE81-6D86-DD11-863A-000423D985E4.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/48659740-6F86-DD11-926F-001617E30D12.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/488BFB15-6F86-DD11-B4A2-000423D6C8EE.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/4AA2ACC0-6D86-DD11-8CC9-001617E30E2C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/5619FDAA-6E86-DD11-85B7-000423D94700.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/5A1FDE7E-6E86-DD11-BAFF-000423D6AF24.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/5A8019BB-6D86-DD11-80F7-001617C3B78C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/64769ED1-6D86-DD11-9684-000423D6CAF2.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/64A80339-6F86-DD11-8266-001617C3B73A.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/66F1CAA9-6E86-DD11-8BE7-000423D992DC.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/6E70BEBC-6D86-DD11-97A0-001617E30CA4.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/7479B5CA-6D86-DD11-8901-001617E30D12.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/7A9AB3C3-6D86-DD11-BCA2-001617E30CC8.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/7E668FC2-6D86-DD11-8260-001617C3B6E8.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/801375AA-6E86-DD11-97AF-000423D9880C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/801C1FA2-6E86-DD11-BAE4-001617C3B69C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/808501C6-6D86-DD11-AB39-001617E30E28.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/80BEEB00-6F86-DD11-8777-0016177CA7A0.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/82C51735-6F86-DD11-8646-001617DBD230.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/8638B0BD-6D86-DD11-99C5-001617DBD230.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/8668573A-6F86-DD11-BB1F-0019DB29C5FC.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/8687E8C2-6D86-DD11-AA19-0019DB2F3F9B.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/86A60FB4-6D86-DD11-B3CD-001617DBD316.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/9A32E433-6F86-DD11-B6BF-001617DF785A.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/9C65C37A-6E86-DD11-A876-000423D6B5C4.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/9E9705BA-6D86-DD11-81E4-001617DC1F70.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/A26163C0-6D86-DD11-984A-001617E30CE8.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/A8CAC58A-6D86-DD11-9805-000423D992A4.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/A8D988C2-6D86-DD11-8342-001617E30F48.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/AA49AFC0-6D86-DD11-9D71-001617DBD5B2.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/AED3FD3C-6F86-DD11-A9F7-001617C3B6CC.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/B2AFB941-6F86-DD11-82E7-001617C3B77C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/BA7B56B9-6D86-DD11-957E-001617E30D38.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/BAA819DC-6E86-DD11-8139-000423D98DB4.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/BC325410-6F86-DD11-9E26-000423D6CAF2.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/C02991CA-6D86-DD11-A4A6-001617E30D06.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/CAF574C0-6D86-DD11-8D8E-001617DBCF90.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/D06ECFC8-6D86-DD11-A19B-001617C3B66C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/D2DB72C0-6D86-DD11-BFBE-001617C3B5F4.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/D4700338-6F86-DD11-9F9D-001617C3B6DC.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/D67B9ABD-6D86-DD11-A6A2-001617DBD288.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/E47BD0AA-6E86-DD11-B89E-000423DD2F34.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/EA298EBC-6D86-DD11-9FAD-001617C3B73A.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/EEE12E83-6D86-DD11-835A-000423D9880C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/F0875383-6E86-DD11-BF96-001617E30F50.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/F0A4EEA2-6E86-DD11-B683-001617C3B710.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/F2A287BF-6D86-DD11-8DD6-001617C3B77C.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/F47E188D-6D86-DD11-B87C-000423D992DC.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/F66E700A-6F86-DD11-B510-000423D6CA42.root',
       '/store/relval/CMSSW_2_1_9/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/FC15908F-6D86-DD11-AF1B-000423D9853C.root')
)

process.dump = cms.EDFilter("EventContentAnalyzer")

process.myanalysis = cms.EDFilter("JPTAnalyzer",
    HistOutFile = cms.untracked.string('analysis.root'),
    calojets = cms.string('iterativeCone5CaloJets'),
    zspjets = cms.string('ZSPJetCorJetIcone5'),
    genjets  = cms.string('iterativeCone5GenJetsNoNuBSM'),
    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorIcone5')
#    genjets  = cms.string('iterativeCone5GenJetsNoNuBSM')
#    genjets = cms.string('iterativeCone5GenJets')
)

from RecoJets.JetAssociationProducers.iterativeCone5JTA_cff import*

ZSPiterativeCone5JetTracksAssociatorAtVertex = iterativeCone5JetTracksAssociatorAtVertex.clone() 
ZSPiterativeCone5JetTracksAssociatorAtVertex.jets = cms.InputTag("ZSPJetCorJetIcone5")

ZSPiterativeCone5JetTracksAssociatorAtCaloFace = iterativeCone5JetTracksAssociatorAtCaloFace.clone()
ZSPiterativeCone5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ZSPJetCorJetIcone5")

ZSPiterativeCone5JetExtender = iterativeCone5JetExtender.clone() 
ZSPiterativeCone5JetExtender.jets = cms.InputTag("ZSPJetCorJetIcone5")
ZSPiterativeCone5JetExtender.jet2TracksAtCALO = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtCaloFace")
ZSPiterativeCone5JetExtender.jet2TracksAtVX = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtVertex")

ZSPrecoJetAssociations = cms.Sequence(ZSPiterativeCone5JetTracksAssociatorAtVertex*ZSPiterativeCone5JetTracksAssociatorAtCaloFace*ZSPiterativeCone5JetExtender)

process.p1 = cms.Path(process.genParticlesForJets*process.iterativeCone5GenJetsNoNuBSM*process.ZSPJetCorrections*process.ZSPrecoJetAssociations*process.myanalysis)

# process.p1 = cms.Path(process.genParticlesForJets*process.iterativeCone5GenJetsNoNuBSM*process.ZSPJetCorrections*process.ZSPrecoJetAssociations*process.dump)

