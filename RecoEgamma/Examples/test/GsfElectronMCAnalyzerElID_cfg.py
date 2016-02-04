import FWCore.ParameterSet.Config as cms

process = cms.Process("readelectrons")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring (
       '/store/relval/CMSSW_2_2_0/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_V9_v1/0000/587EC8EF-B4B9-DD11-AC52-001617C3B65A.root',
       '/store/relval/CMSSW_2_2_0/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_V9_v1/0000/9E464300-76B9-DD11-B526-000423D98C20.root'
    )
)

process.gsfElectronAnalysis = cms.EDAnalyzer("GsfElectronMCAnalyzerElID",
    electronCollection = cms.InputTag("gsfElectrons"),
    eidValueMapName = cms.string("eidLoose"),
    #eidValueMapName = cms.string("eidRobustLoose"),
    #eidValueMapName = cms.string("eidTight"),
    #eidValueMapName = cms.string("eidRobustTight"),
    mcTruthCollection = cms.InputTag("source"),
    outputFile = cms.string('gsfElectronHistos_mc_elID.root'),
    MaxPt = cms.double(100.0),
    DeltaR = cms.double(0.05),
    MaxAbsEta = cms.double(2.5),
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
    Fhitsmax = cms.double(20.0),
    Lhitsmax = cms.double(10.0),
    Nbinxyz = cms.int32(50),
    Nbineop2D = cms.int32(30),
    Nbinp = cms.int32(50),
    Nbineta2D = cms.int32(50),
    Nbinfhits = cms.int32(20),
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
    Nbineop = cms.int32(50)
)

process.p = cms.Path(process.gsfElectronAnalysis)


