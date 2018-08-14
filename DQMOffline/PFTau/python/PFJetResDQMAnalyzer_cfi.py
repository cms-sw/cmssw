import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
pfJetResDQMAnalyzer = DQMEDAnalyzer('PFJetDQMAnalyzer',
    InputCollection = cms.InputTag('pfAllElectrons'),
    MatchCollection = cms.InputTag('gensource'),
    BenchmarkLabel  = cms.string('PFJetResValidation/PFElecVsGenElec'),
    deltaRMax = cms.double(0.1),
    onlyTwoJets = cms.bool(True),
    matchCharge = cms.bool(False),
    mode = cms.int32( 1 ),
    CreatePFractionHistos = cms.bool(False),
    CreateReferenceHistos = cms.bool(True),
    CreateEfficiencyHistos = cms.bool(False),
    ptMin = cms.double( 0.0 ),
    ptMax = cms.double( 999999 ),
    etaMin = cms.double(-10),
    etaMax = cms.double(10),
    phiMin = cms.double(-3.14),
    phiMax = cms.double(3.14),
# Histogram Parameters related to pt
    #VariablePtBins  = cms.vdouble(0.,1.,2.,5.,10.,20.,50.,100.,200.,400.,1000.),
    VariablePtBins  = cms.vdouble(20,40,60,80,100,150,200,250,300,400,500,750), # must be = to the one in PFClient_cfi if you want to slice the TH2
    PtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(50),
      xMin = cms.double(0.0),
      xMax = cms.double(100.0)        
    ),
    DeltaPtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(100),
      xMin = cms.double(-50.0),
      xMax = cms.double(50.0)        
    ),
    DeltaPtOvPtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      BROn = cms.bool(True), BREtaMin = cms.double(0.0), BREtaMax = cms.double(1.4),
      EROn = cms.bool(True), EREtaMin = cms.double(1.6), EREtaMax = cms.double(2.4),
      slicingOn = cms.bool(False),
      nBin = cms.int32(160), #200
      xMin = cms.double(-1.0),
      xMax = cms.double(1.0)        
    ),
# Histogram Parameters related to Eta                               
    EtaHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(100),
      xMin = cms.double(-5.0),
      xMax = cms.double(5.0)        
    ),
    DeltaEtaHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(50),
      xMin = cms.double(-0.5),
      xMax = cms.double(0.5)        
    ),
# Histogram Parameters related to Phi                               
    PhiHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(100),
      xMin = cms.double(-3.1416),
      xMax = cms.double(3.1416)        
    ),
    DeltaPhiHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(50),
      xMin = cms.double(-0.5),
      xMax = cms.double(0.5)        
    ),
# Histogram Parameters related to DeltaR     
   DeltaRHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(150), 
      xMin = cms.double(0.0),
      xMax = cms.double(1.0)        
    ),
# Histogram Parameters related to Charge                               
    ChargeHistoParameter = cms.PSet(
      switchOn = cms.bool(False),
      nBin = cms.int32(3),
      xMin = cms.double(-1.5),
      xMax = cms.double(1.5)        
    )
)
