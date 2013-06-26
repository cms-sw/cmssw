import FWCore.ParameterSet.Config as cms

pfMETDQMAnalyzer = cms.EDAnalyzer("PFMETDQMAnalyzer",
    InputCollection = cms.InputTag('pfMet'),
    MatchCollection = cms.InputTag('met'),
    BenchmarkLabel  = cms.string('PFMET/CompWithCaloMET'),
    mode            = cms.int32( 1 ),
    CreateMETSpecificHistos = cms.bool(True),
    ptMin = cms.double( 0.0 ),
    ptMax = cms.double( 999999 ),
    etaMin = cms.double(-10),
    etaMax = cms.double(10),
    phiMin = cms.double(-3.14),
    phiMax = cms.double(3.14),
# Histogram Parameters related to pt
    VariablePtBins  = cms.vdouble(0.,1.,2.,5.,10.,15.,20.,30.,50.,100.,500.),
    PtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(100),
      xMin = cms.double(0.0),
      xMax = cms.double(200.0)        
    ),
    DeltaPtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(50),
      xMin = cms.double(-300.0),
      xMax = cms.double(300.0)        
    ),
    DeltaPtOvPtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(100),
      xMin = cms.double(-3.0),
      xMax = cms.double(3.0)        
    ),
# Histogram Parameters related to Eta                               
    EtaHistoParameter = cms.PSet(
      switchOn = cms.bool(False),
      nBin = cms.int32(100),
      xMin = cms.double(-5.0),
      xMax = cms.double(5.0)        
    ),
    DeltaEtaHistoParameter = cms.PSet(
      switchOn = cms.bool(False),
      nBin = cms.int32(50),
      xMin = cms.double(-0.2),
      xMax = cms.double(0.2)        
    ),
# Histogram Parameters related to Phi                               
    PhiHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(64),
      xMin = cms.double(-3.2),
      xMax = cms.double(3.2)        
    ),
    DeltaPhiHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(64),
      xMin = cms.double(-3.2),
      xMax = cms.double(3.2)        
    ),
# Histogram Parameters related to Px and Py
    PxHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(50),
      xMin = cms.double(0.0),
      xMax = cms.double(200.0)        
    ),
    DeltaPxHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(50),
      xMin = cms.double(-300.0),
      xMax = cms.double(300.0)        
    ),
# Histogram Parameters related to Sum Et
    SumEtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(100),
      xMin = cms.double(0.0),
      xMax = cms.double(1000.0)        
    ),
    DeltaSumEtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(100),
      xMin = cms.double(-300.0),
      xMax = cms.double(500.0)        
    ),
    DeltaSumEtOvSumEtHistoParameter = cms.PSet(
      switchOn = cms.bool(True),
      nBin = cms.int32(100),
      xMin = cms.double(-1.0),
      xMax = cms.double(3.0),        
    ),
# Histogram Parameters related to Charge                               
    ChargeHistoParameter = cms.PSet(
      switchOn = cms.bool(False),
      nBin = cms.int32(3),
      xMin = cms.double(-1.5),
      xMax = cms.double(1.5)        
    ),
# parameter for event skim
    SkimParameter = cms.PSet(
      switchOn = cms.bool(False),
      maximumNumberToBeStored = cms.int32(100),
      lowerCutOffOnResolution = cms.double(-1.5),
      upperCutOffOnResolution = cms.double(1.5)
    )
)
