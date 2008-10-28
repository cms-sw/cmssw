import FWCore.ParameterSet.Config as cms

process = cms.Process("readelectrons")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring (
       '/store/relval/CMSSW_2_1_7/RelValSingleElectronPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/6AC34254-A77D-DD11-A69E-000423D99B3E.root',
       '/store/relval/CMSSW_2_1_7/RelValSingleElectronPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/CEF2036E-A77D-DD11-8C3E-001617DBD556.root',
       '/store/relval/CMSSW_2_1_7/RelValSingleElectronPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/DA54377C-A77D-DD11-B415-000423D98C20.root',
       '/store/relval/CMSSW_2_1_7/RelValSingleElectronPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0002/F6AD9822-437E-DD11-8D4C-001617DBD332.root'
    )
)

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
    outputFile = cms.string('gsfElectronHistos_RelVal2110SinglePiPt1.root'),
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


