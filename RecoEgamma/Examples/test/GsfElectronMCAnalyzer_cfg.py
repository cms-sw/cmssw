import FWCore.ParameterSet.Config as cms

process = cms.Process("readelectrons")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring (
    '/store/relval/2008/7/21/RelVal-RelValSingleElectronPt35-1216579481-IDEAL_V5-2nd/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/0808CDA5-6E57-DD11-9CC1-000423D99896.root',
    '/store/relval/2008/7/21/RelVal-RelValSingleElectronPt35-1216579481-IDEAL_V5-2nd/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/1AEAE04D-6E57-DD11-AAA9-000423D94AA8.root',
    '/store/relval/2008/7/21/RelVal-RelValSingleElectronPt35-1216579481-IDEAL_V5-2nd/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/B6322A11-6E57-DD11-8A69-000423D99264.root'
    )
)
#process.PoolSource.fileNames = [
#'/store/relval/2008/7/21/RelVal-RelValSingleElectronPt10-1216579481-IDEAL_V5-2nd/RelValSingleElectronPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/34931F15-6E57-DD11-8E18-001617E30CD4.root',
#'/store/relval/2008/7/21/RelVal-RelValSingleElectronPt10-1216579481-IDEAL_V5-2nd/RelValSingleElectronPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/8C715432-6D57-DD11-9BDD-000423D991F0.root',
#'/store/relval/2008/7/21/RelVal-RelValSingleElectronPt10-1216579481-IDEAL_V5-2nd/RelValSingleElectronPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/DCB702F2-6D57-DD11-B9E7-000423D6AF24.root'
#]
process.PoolSource.fileNames = [
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/18F42589-6D57-DD11-A9A5-000423D6BA18.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/36BF874C-6E57-DD11-BD9B-001617C3B6E8.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/3E82ACE9-6D57-DD11-9A0D-001617C3B6CC.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/40A430AF-6E57-DD11-9B2C-001617C3B77C.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/5277E47A-6E57-DD11-B0B6-000423D98BE8.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/5C0EA24E-6E57-DD11-BE94-000423D99896.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/68DCC988-6E57-DD11-8AC3-000423D98834.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/6C5BB146-6E57-DD11-9B5E-000423D9870C.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/84BCF673-6E57-DD11-A56E-000423D94AA8.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/88B626A5-6E57-DD11-B5B4-001617C3B70E.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/8AC97AA5-6E57-DD11-9945-000423D6BA18.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/A46D60A2-6E57-DD11-A43E-001617E30F56.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/B2858DE6-6D57-DD11-94AD-000423D99264.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/B6303EEA-6E57-DD11-B2B4-000423D992DC.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/D807B75A-6E57-DD11-A5AC-000423D99996.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/E8DAC9A0-6E57-DD11-95AA-000423D6C8EE.root',
'/store/relval/2008/7/21/RelVal-RelValZEE-1216579576-STARTUP_V4-2nd/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579576-STARTUP_V4-2nd-unmerged/0000/FA33055B-6D57-DD11-82B6-001617C3B65A.root'
]
process.gsfElectronAnalysis = cms.EDAnalyzer("GsfElectronMCAnalyzer",
    electronCollection = cms.InputTag("pixelMatchGsfElectrons"),
    Nbinxyz = cms.int32(50),
    Nbineop2D = cms.int32(30),
    Nbinp = cms.int32(50),
    Nbineta2D = cms.int32(50),
    Etamin = cms.double(-2.5),
    Nbinfhits = cms.int32(20),
    Dphimin = cms.double(-0.01),
    Pmax = cms.double(300.0),
    Phimax = cms.double(3.2),
    Phimin = cms.double(-3.2),
    Eopmax = cms.double(5.0),
    mcTruthCollection = cms.InputTag("source"),
    MaxPt = cms.double(100.0),
    Nbinlhits = cms.int32(5),
    Nbinpteff = cms.int32(19),
    Nbinphi2D = cms.int32(32),
    Nbindetamatch2D = cms.int32(50),
    Nbineta = cms.int32(50),
    DeltaR = cms.double(0.05),
    outputFile = cms.string('gsfElectronHistos.root'),
    Nbinp2D = cms.int32(50),
    Nbindeta = cms.int32(100),
    Nbinpt2D = cms.int32(50),
    Nbindetamatch = cms.int32(100),
    Fhitsmax = cms.double(20.0),
    Lhitsmax = cms.double(10.0),
    Nbinphi = cms.int32(64),
    Eopmaxsht = cms.double(3.0),
    MaxAbsEta = cms.double(2.5),
    Nbindphimatch = cms.int32(100),
    Detamax = cms.double(0.005),
    Nbinpt = cms.int32(50),
    Nbindphimatch2D = cms.int32(50),
    Etamax = cms.double(2.5),
    Dphimax = cms.double(0.01),
    Dphimatchmax = cms.double(0.2),
    Detamatchmax = cms.double(0.05),
    Nbindphi = cms.int32(100),
    Detamatchmin = cms.double(-0.05),
    Ptmax = cms.double(100.0),
    Nbineop = cms.int32(50),
    Dphimatchmin = cms.double(-0.2),
    Detamin = cms.double(-0.005)
)

process.p = cms.Path(process.gsfElectronAnalysis)


