
import FWCore.ParameterSet.Config as cms

fakeAnalyzerFineBiningParameters = cms.PSet(
    Nbinxyz = cms.int32(200), 
    Nbinp = cms.int32(300), Nbinp2D = cms.int32(100), Pmax = cms.double(300.0),
    Nbinfhits = cms.int32(30), Fhitsmax = cms.double(30.0),
    Nbinlhits = cms.int32(5), Lhitsmax = cms.double(10.0),
    Nbineta = cms.int32(250), Nbineta2D = cms.int32(100), Etamin = cms.double(-2.5), Etamax = cms.double(2.5),
    Nbindeta = cms.int32(300), Detamin = cms.double(-0.005), Detamax = cms.double(0.005),
    Nbindetamatch = cms.int32(200), Nbindetamatch2D = cms.int32(100), Detamatchmin = cms.double(-0.05), Detamatchmax = cms.double(0.05),
    Nbinphi = cms.int32(128), Nbinphi2D = cms.int32(128), Phimin = cms.double(-3.2), Phimax = cms.double(3.2),
    Nbindphimatch = cms.int32(300), Nbindphimatch2D = cms.int32(100), Dphimatchmin = cms.double(-0.2), Dphimatchmax = cms.double(0.2),
    Nbinpt = cms.int32(300), Nbinpt2D = cms.int32(100), Nbinpteff = cms.int32(190), Ptmax = cms.double(100.0),
    Nbindphi = cms.int32(300), Dphimin = cms.double(-0.01), Dphimax = cms.double(0.01),
    Nbineop = cms.int32(300), Nbineop2D = cms.int32(100), Eopmax = cms.double(5.0), Eopmaxsht = cms.double(3.0),
    Nbinmee = cms.int32(300), Meemin = cms.double(0.0), Meemax = cms.double(150.),
    Nbinhoe = cms.int32(200), Hoemin = cms.double(0.0), Hoemax = cms.double(0.5)
)
