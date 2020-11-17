# The following comments couldn't be translated into the new config version:

# histos limits and binning

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/6/RelVal-RelValSingleElectronPt35-1212531852-IDEAL_V1-2nd-02/0000/4202AC1C-EA33-DD11-9082-001617DC1F70.root', 
        '/store/relval/2008/6/6/RelVal-RelValSingleElectronPt35-1212531852-IDEAL_V1-2nd-02/0000/6206F77E-EB33-DD11-9B1B-001617DBD5B2.root', 
        '/store/relval/2008/6/6/RelVal-RelValSingleElectronPt35-1212531852-IDEAL_V1-2nd-02/0000/86842A33-E933-DD11-887F-001617C3B5F4.root')
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
    # efficiency cuts
    MaxPt = cms.double(100.0),
    Nbinlhits = cms.int32(5),
    Nbinpteff = cms.int32(19),
    Nbinphi2D = cms.int32(32),
    Nbindetamatch2D = cms.int32(50),
    Nbineta = cms.int32(50),
    DeltaR = cms.double(0.05),
    outputFile = cms.string('gsfElectronHistos-full.root'),
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

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.gsfElectronAnalysis)
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.detailedInfo = dict(extension='txt')


