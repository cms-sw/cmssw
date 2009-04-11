# The following comments couldn't be translated into the new config version:

# histos limits and binning

import FWCore.ParameterSet.Config as cms

gsfElectronAnalysis = cms.EDAnalyzer("ElectronAnalyzer",
    electronCollection = cms.InputTag("gsfElectrons"),
    OutputMEsInRootFile = cms.bool(False),
    Nbinxyz = cms.int32(50),
    Nbineop2D = cms.int32(30),
    Nbinp = cms.int32(50),
    Lhitsmax = cms.double(10.0),
    Etamin = cms.double(-2.5),
    Nbinfhits = cms.int32(30),
    Dphimin = cms.double(-0.01),
    Pmax = cms.double(300.0),
    Phimax = cms.double(3.2),
    Phimin = cms.double(-3.2),
    Eopmax = cms.double(5.0),
    # efficiency cuts
    MaxPt = cms.double(100.0),
    Nbinlhits = cms.int32(5),
    Nbinpteff = cms.int32(19),
    Nbinphi2D = cms.int32(32),
    Nbindetamatch2D = cms.int32(50),
    Nbineta = cms.int32(50),
    DeltaR = cms.double(0.05),
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    outputFile = cms.string('gsfElectronHistos.root'),
    Nbinp2D = cms.int32(50),
    Nbindeta = cms.int32(100),
    # DBE verbosity
    Verbosity = cms.untracked.int32(2),
    Nbinpt2D = cms.int32(50),
    Nbindetamatch = cms.int32(100),
    Fhitsmax = cms.double(30.0),
    matchingObjectCollection = cms.InputTag("mergedSuperClusters"),
    Nbinphi = cms.int32(64),
    Nbineta2D = cms.int32(50),
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


