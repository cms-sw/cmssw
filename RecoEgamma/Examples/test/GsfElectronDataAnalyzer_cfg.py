import FWCore.ParameterSet.Config as cms

process = cms.Process("readelectrons")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring (
        '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0008/FE09D3D8-4157-DE11-8414-001D09F24934.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0008/FA22A5E6-4257-DE11-BA8A-001D09F24259.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0008/D024F0B4-4257-DE11-850E-001D09F24E39.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0008/CA089AA6-0458-DE11-9CF1-001D09F290BF.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0008/BA623C75-4057-DE11-B8D3-001D09F29849.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0008/963CA01D-4257-DE11-B6FD-001617E30F48.root'
    )
)

process.mergedSuperClusters = cms.EDFilter("EgammaSuperClusterMerger",
    src = cms.VInputTag(cms.InputTag("correctedHybridSuperClusters"), cms.InputTag("multi5x5SuperClustersWithPreshower"))
)

process.gsfElectronAnalysis = cms.EDAnalyzer("GsfElectronDataAnalyzer",
    electronCollection = cms.InputTag("gsfElectrons"),
    MinEt = cms.double(4.),
    MinPt = cms.double(0.),
    MaxAbsEta = cms.double(2.5),
    SelectEB = cms.bool(False),
    SelectEE = cms.bool(False),
    SelectNotEBEEGap = cms.bool(False),
    SelectEcalDriven = cms.bool(False),
    SelectTrackerDriven = cms.bool(False),
    MinEOverPBarrel = cms.double(0.),
    MaxEOverPBarrel = cms.double(10000.),
    MinEOverPEndcaps = cms.double(0.),
    MaxEOverPEndcaps = cms.double(10000.),
    MinDetaBarrel = cms.double(0.),
    MaxDetaBarrel = cms.double(10000.),
    MinDetaEndcaps = cms.double(0.),
    MaxDetaEndcaps = cms.double(10000.),
    MinDphiBarrel = cms.double(0.),
    MaxDphiBarrel = cms.double(10000.),
    MinDphiEndcaps = cms.double(0.),
    MaxDphiEndcaps = cms.double(10000.),
    MinSigIetaIetaBarrel = cms.double(0.),
    MaxSigIetaIetaBarrel = cms.double(10000.),
    MinSigIetaIetaEndcaps = cms.double(0.),
    MaxSigIetaIetaEndcaps = cms.double(10000.),
    MaxHoEBarrel = cms.double(10000.),
    MaxHoEEndcaps = cms.double(10000.),
    MinMVA = cms.double(-10000.),
    MaxTipBarrel = cms.double(10000.),
    MaxTipEndcaps = cms.double(10000.),
    MaxTkIso03 = cms.double(1.0),
    MaxHcalIso03Depth1Barrel = cms.double(10000.),
    MaxHcalIso03Depth1Endcaps = cms.double(10000.),
    MaxHcalIso03Depth2Endcaps = cms.double(10000.),
    MaxEcalIso03Barrel = cms.double(10000.),
    MaxEcalIso03Endcaps = cms.double(10000.),
    matchingObjectCollection = cms.InputTag("mergedSuperClusters"),
    MaxPt = cms.double(100.0),
    DeltaR = cms.double(0.3),
    outputFile = cms.string('gsfElectronHistos_data_RelVal310pre10QCD_Pt_80_120_tkIso.root'),
    Etamin = cms.double(-2.5),
    Etamax = cms.double(2.5),
    Phimax = cms.double(3.2),
    Phimin = cms.double(-3.2),
    Ptmax = cms.double(100.0),
    Pmax = cms.double(300.0),
    Eopmax = cms.double(5.0),
    Eopmaxsht = cms.double(3.0),
    Detamin = cms.double(-0.005),
    Detamax = cms.double(0.005),
    Dphimin = cms.double(-0.01),
    Dphimax = cms.double(0.01),
    Dphimatchmin = cms.double(-0.2),
    Dphimatchmax = cms.double(0.2),
    Detamatchmin = cms.double(-0.05),
    Detamatchmax = cms.double(0.05),
    Fhitsmax = cms.double(30.0),
    Lhitsmax = cms.double(10.0),
    Nbinxyz = cms.int32(50),
    Nbineop2D = cms.int32(30),
    Nbinp = cms.int32(50),
    Nbineta2D = cms.int32(50),
    Nbinfhits = cms.int32(30),
    Nbinlhits = cms.int32(5),
    Nbinpteff = cms.int32(19),
    Nbinphi2D = cms.int32(32),
    Nbindetamatch2D = cms.int32(50),
    Nbineta = cms.int32(50),
    Nbinp2D = cms.int32(50),
    Nbindeta = cms.int32(100),
    Nbinpt2D = cms.int32(50),
    Nbindetamatch = cms.int32(100),
    Nbinphi = cms.int32(64),
    Nbindphimatch = cms.int32(100),
    Nbinpt = cms.int32(50),
    Nbindphimatch2D = cms.int32(50),
    Nbindphi = cms.int32(100),
    Nbineop = cms.int32(50),
    Nbinpoptrue = cms.int32(75),
    Poptruemin = cms.double(0.0),
    Poptruemax = cms.double(1.5),
    Nbinmee = cms.int32(100),
    Meemin = cms.double(0.0),
    Meemax = cms.double(150.)
    Nbinhoe = cms.int32(100),
    Hoemin = cms.double(0.0),
    Hoemax = cms.double(0.5)
)

process.p = cms.Path(process.mergedSuperClusters*process.gsfElectronAnalysis)


