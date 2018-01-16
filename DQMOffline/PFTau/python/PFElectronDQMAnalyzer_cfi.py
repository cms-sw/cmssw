import FWCore.ParameterSet.Config as cms

pfElectronDQMAnalyzer = DQMStep1Module('PFCandidateDQMAnalyzer',
    InputCollection = cms.InputTag('pfAllElectrons'),
    MatchCollection = cms.InputTag('gensource'),
    BenchmarkLabel  = cms.string('PFElectronValidation/PFElecVsGenElec'),
    deltaRMax = cms.double(0.2),
    matchCharge = cms.bool(True),
    mode = cms.int32( 1 ),
    CreateReferenceHistos = cms.bool(True),
    CreateEfficiencyHistos = cms.bool(True),
    ptMin = cms.double( 2.0 ), # since pT_reco seem to have this threshold
    ptMax = cms.double( 999999 ),
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
    phiMin = cms.double(-3.14),
    phiMax = cms.double(3.14),
    slimmedLikeSelection = cms.bool(False),
    # Histogram Parameters related to pt
    #VariablePtBins  = cms.vdouble(0.,1.,2.,5.,10.,20.,50.,100.,200.,400.,1000.),
    VariablePtBins  = cms.vdouble(0.), # if only one entry PtHistoParameter used
    PtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(60),
      xMin = cms.double(0.0),
      xMax = cms.double(120.0)        
    ),
    DeltaPtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(100),
      xMin = cms.double(-30.0),
      xMax = cms.double(30.0)        
    ),
    DeltaPtOvPtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      BROn = cms.bool(False), BREtaMin = cms.double(0.0), BREtaMax = cms.double(1.4),
      EROn = cms.bool(False), EREtaMin = cms.double(1.6), EREtaMax = cms.double(2.4),
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
      nBin = cms.int32(400),
      xMin = cms.double(-0.2),
      xMax = cms.double(0.2)        
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
      nBin = cms.int32(400), 
      xMin = cms.double(-0.2),
      xMax = cms.double(0.2)        
    ),
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
