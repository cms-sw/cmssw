import FWCore.ParameterSet.Config as cms


photonAnalysis = cms.EDAnalyzer("PhotonAnalyzer",

    Name = cms.untracked.string('photonAnalysis'),

    phoProducer = cms.string('photons'),
    photonCollection = cms.string(''),


    triggerEvent = cms.InputTag("hltTriggerSummaryAOD",""),                            
    prescaleFactor = cms.untracked.int32(1),

    useBinning = cms.bool(False),
    useTriggerFiltering = cms.bool(True),                             
    standAlone = cms.bool(False),
                                
    minPhoEtCut = cms.double(0.0),
              
    cutStep = cms.double(50.0),
    numberOfSteps = cms.int32(2),
                                
    # DBE verbosity
    Verbosity = cms.untracked.int32(0),
                                # 1 provides basic output
                                # 2 provides output of the fill step + 1
                                # 3 provides output of the store step + 2
                                
    isolationStrength = cms.int32(0),
                                # 0 => Loose Photon
                                # 1 => Tight Photon



    ##### Histogram Ranges and Bins                               

    eBin = cms.int32(200),
    eMin = cms.double(0.0),
    eMax = cms.double(500.0),
                                
    etBin = cms.int32(200),
    etMin = cms.double(0.0),
    etMax = cms.double(200.0),

    sumBin = cms.int32(200),
    sumMin = cms.double(0.0),
    sumMax = cms.double(20.0),
                                
    etaBin = cms.int32(200),                               
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
    barrelEtaBin = cms.int32(170),                               
    barrelEtaMin = cms.double(-1.5),
    barrelEtaMax = cms.double(1.5),                                

    phiBin = cms.int32(200),
    barrelPhiBin = cms.int32(360),                               
    phiMin = cms.double(-3.14),
    phiMax = cms.double(3.14),
                                
    r9Bin = cms.int32(110),
    r9Min = cms.double(0.0),
    r9Max = cms.double(1.1),

    hOverEBin = cms.int32(200),
    hOverEMin = cms.double(0),                               
    hOverEMax = cms.double(0.5), 

    numberBin = cms.int32(10),
    numberMin = cms.double(0),                               
    numberMax = cms.double(10), 

    xyBin = cms.int32(100),
    xyMin = cms.double(-150),                               
    xyMax = cms.double(150),

    rBin = cms.int32(200),
    rMin = cms.double(0),                               
    rMax = cms.double(120),                                

    zBin = cms.int32(100),
    zMin = cms.double(0),                               
    zMax = cms.double(280),

    dEtaTracksBin = cms.int32(100),
    dEtaTracksMin = cms.double(-0.2),
    dEtaTracksMax = cms.double(0.2),

    dPhiTracksBin = cms.int32(100),
    dPhiTracksMin = cms.double(-0.5),
    dPhiTracksMax = cms.double(0.5),

    dRBin = cms.int32(300),
    dRMin = cms.double(0.0),
    dRMax = cms.double(0.1),

    sigmaIetaBin = cms.int32(200),
    sigmaIetaMin = cms.double(0.0),
    sigmaIetaMax = cms.double(0.05),

    eOverPBin = cms.int32(100),
    eOverPMin = cms.double(0.0),
    eOverPMax = cms.double(5.0),                                

    ######
                                
                
    OutputMEsInRootFile = cms.bool(False),
 
    OutputFileName = cms.string('DQMOfflinePhotons.root'),


)
