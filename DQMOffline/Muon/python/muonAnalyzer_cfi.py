import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

# MuonAnalyzer
muonAnalyzer = cms.EDAnalyzer("MuonAnalyzer",
                              MuonServiceProxy,
                              OutputMEsInRootFile = cms.bool(False),
                              trackSegmentsAnalysis = cms.PSet(    
    phiMin = cms.double(-3.2),
    ptBin = cms.int32(200),
    SegmentsTrackAssociatorParameters = cms.PSet(
    segmentsDt = cms.untracked.InputTag("dt4DSegments"),
    SelectedSegments = cms.untracked.InputTag("SelectedSegments"),
    segmentsCSC = cms.untracked.InputTag("cscSegments")
    ),
    etaBin = cms.int32(100),
    etaMin = cms.double(-3.0),
    ptMin = cms.double(0.0),
    phiBin = cms.int32(100),
    ptMax = cms.double(200.0),
    etaMax = cms.double(3.0),
    phiMax = cms.double(3.2)
    ),
                              GlobalMuTrackCollection = cms.InputTag("globalMuons"),
                              SeedCollection = cms.InputTag("ancientMuonSeed"),
                              muonRecoAnalysis = cms.PSet(
    thetaMin = cms.double(0.0),
    phiMin = cms.double(-3.2),
    chi2Min = cms.double(0),
    ptBin = cms.int32(500),
    thetaBin = cms.int32(100),
    rhBin = cms.int32(25),
    pResMin = cms.double(-0.01),
    pResMax = cms.double(0.01),
    thetaMax = cms.double(3.2),
    pResBin = cms.int32(50),
    rhMin = cms.double(0.0),
    pMin = cms.double(0.0),
    rhMax = cms.double(1.001),
    etaMin = cms.double(-3.0),
    etaBin = cms.int32(100),
    phiBin = cms.int32(100),
    chi2Bin = cms.int32(100),
    pBin = cms.int32(500),
    ptMin = cms.double(0.0),
    ptMax = cms.double(500.0),
    etaMax = cms.double(3.0),
    pMax = cms.double(500.0),
    phiMax = cms.double(3.2),
    chi2Max = cms.double(20),
    tunePBin = cms.int32(100),
    tunePMin = cms.double(-1.0),
    tunePMax = cms.double(1.0)
    ),
                              DoMuonSeedAnalysis = cms.untracked.bool(True),
                              DoTrackSegmentsAnalysis = cms.untracked.bool(True),
                              seedsAnalysis = cms.PSet(
    seedPxyzMin = cms.double(-50.0),
    pxyzErrMin = cms.double(-100.0),
    phiErrMax = cms.double(3.2),
    pxyzErrMax = cms.double(100.0),
    RecHitBin = cms.int32(25),
    etaErrMin = cms.double(0.0),
    seedPtMin = cms.double(0.0),
    seedPxyzBin = cms.int32(100),
    ThetaBin = cms.int32(100),
    RecHitMin = cms.double(0.0),
    EtaMin = cms.double(-3.0),
    pErrBin = cms.int32(200),
    phiErrBin = cms.int32(160),
    EtaMax = cms.double(3.0),
    etaErrBin = cms.int32(200),
    seedPxyzMax = cms.double(50.0),
    ThetaMin = cms.double(0.0),
    PhiMin = cms.double(-3.2),
    pxyzErrBin = cms.int32(100),
    RecHitMax = cms.double(25.0),
    ThetaMax = cms.double(3.2),
    pErrMin = cms.double(0.0),
    EtaBin = cms.int32(100),
    pErrMax = cms.double(200.0),
    seedPtMax = cms.double(200.0),
    seedPtBin = cms.int32(1000),
    phiErrMin = cms.double(0.0),
    PhiBin = cms.int32(100),
    debug = cms.bool(False),
    etaErrMax = cms.double(0.5),
    PhiMax = cms.double(3.2)
    ),
                              OutputFileName = cms.string('MuonMonitoring.root'),
                              DoMuonEnergyAnalysis = cms.untracked.bool(True),
                              STAMuTrackCollection = cms.InputTag("standAloneMuons"),
                              DoMuonRecoAnalysis = cms.untracked.bool(True),
                              MuonCollection = cms.InputTag("muons"),
                              TriggerResultsLabel = cms.InputTag("TriggerResults::HLT"),
                              muonEnergyAnalysis = cms.PSet(
    AlgoName = cms.string('muons'),
    hadS9SizeMin = cms.double(0.0), 
    emSizeMin = cms.double(0.0),
    emS9SizeBin = cms.int32(100),
    emS9SizeMin = cms.double(0.0),
    hoSizeMax = cms.double(4.0),
    hoS9SizeBin = cms.int32(100),
    hoSizeMin = cms.double(0.0),
    emSizeMax = cms.double(4.0),
    hadS9SizeMax = cms.double(10.0),
    hoS9SizeMin = cms.double(0.0),
    hadSizeMin = cms.double(0.0),
    emSizeBin = cms.int32(100),
    hadS9SizeBin = cms.int32(200),
    debug = cms.bool(False),
    emS9SizeMax = cms.double(4.0),
    hoS9SizeMax = cms.double(4.0),
    hadSizeMax = cms.double(10.0),
    hoSizeBin = cms.int32(100),
    hadSizeBin = cms.int32(200)
    ),
                              DoMuonKinVsEtaAnalysis = cms.untracked.bool(True),                           
                              muonKinVsEtaAnalysis = cms.PSet(

    vertexLabel     = cms.InputTag("offlinePrimaryVertices"),
    bsLabel         = cms.InputTag("offlineBeamSpot"),
     
    pBin = cms.int32(100),
    pMin = cms.double(0.0),
    pMax = cms.double(100.0),
    
    ptBin = cms.int32(100),
    ptMin = cms.double(0.0),
    ptMax = cms.double(100.0),
    
    etaBin = cms.int32(100),
    etaMin = cms.double(-3.0),
    etaMax = cms.double(3.0),

    phiBin = cms.int32(100),
    phiMin = cms.double(-3.2),
    phiMax = cms.double(3.2),

    chiBin = cms.int32(100),
    chiMin = cms.double(0.),
    chiMax = cms.double(20.),

    chiprobMin = cms.double(0.),
    chiprobMax = cms.double(1.),
    
    etaBMin = cms.double(0.),
    etaBMax = cms.double(1.1),
    etaECMin = cms.double(0.9),
    etaECMax = cms.double(2.4),
    etaOvlpMin = cms.double(0.9),
    etaOvlpMax = cms.double(1.1)
    ),
                              DoMuonRecoOneHLT = cms.untracked.bool(True),                           
                              muonRecoOneHLTAnalysis = cms.PSet(
    MuonCollection = cms.InputTag("muons"),
    vertexLabel     = cms.InputTag("offlinePrimaryVertices"),
    bsLabel         = cms.InputTag("offlineBeamSpot"),
    
    ptBin = cms.int32(50),
    ptMin = cms.double(0.0),
    ptMax = cms.double(100.0),
    
    etaBin = cms.int32(50),
    etaMin = cms.double(-3.0),
    etaMax = cms.double(3.0),
    
    phiBin = cms.int32(50),
    phiMin = cms.double(-3.2),
    phiMax = cms.double(3.2),

    chi2Bin = cms.int32(50),
    chi2Min = cms.double(0.),
    chi2Max = cms.double(20),
    
    SingleMuonTrigger = cms.PSet(
      andOr         = cms.bool( True ),
      dbLabel        = cms.string( "MuonDQMTrigger"),
      hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
      hltDBKey       = cms.string('SingleMu'),
      hltPaths       = cms.vstring( "HLT_IsoMu30_eta2p1_v7" ),
      andOrHlt       = cms.bool( True ),
      errorReplyHlt  = cms.bool( False ),
      ),
    DoubleMuonTrigger = cms.PSet(
      andOr         = cms.bool( True ),
      dbLabel        = cms.string( "MuonDQMTrigger"),
      hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
      hltDBKey       = cms.string('DoubleMu'),
      hltPaths       = cms.vstring( "HLT_Mu17_Mu8_v11" ),
      andOrHlt       = cms.bool( True ),
      errorReplyHlt  = cms.bool( False ),
      )
    ),
                              DoDiMuonHistograms = cms.untracked.bool(True),
                              dimuonHistograms = cms.PSet(
    MuonCollection = cms.InputTag("muons"),
    vertexLabel     = cms.InputTag("offlinePrimaryVertices"),
    bsLabel         = cms.InputTag("offlineBeamSpot"),


    etaBin = cms.int32(400),
    etaBBin = cms.int32(400),
    etaEBin = cms.int32(200),
    
    etaBMin = cms.double(0.),
    etaBMax = cms.double(1.1),
    etaECMin = cms.double(0.9),
    etaECMax = cms.double(2.4),
    
    LowMassMin = cms.double(2.0),
    LowMassMax = cms.double(55.0),
    HighMassMin = cms.double(55.0),
    HighMassMax = cms.double(155.0)
    ),
                              DoEfficiencyAnalysis = cms.untracked.bool(True),
                              efficiencyAnalysis = cms.PSet(
    MuonCollection = cms.InputTag("muons"),
    TrackCollection = cms.InputTag("generalTracks"),

    doPrimaryVertexCheck = cms.bool( True ),
    vertexLabel     = cms.InputTag("offlinePrimaryVertices"),
    bsLabel         = cms.InputTag("offlineBeamSpot"),

    ptBin = cms.int32(10),
    ptMax = cms.double(100),
    ptMin = cms.double(10),
    
    etaBin = cms.int32(8),
    etaMax = cms.double(2.5),
    etaMin = cms.double(-2.5),

    phiBin = cms.int32(8),
    phiMax = cms.double(3.2),
    phiMin = cms.double(-3.2),

    vtxBin = cms.int32(10),
    vtxMax = cms.double(40.5),
    vtxMin = cms.double(0.5)
    

    )
                              
                              )




