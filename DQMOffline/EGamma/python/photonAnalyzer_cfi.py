import FWCore.ParameterSet.Config as cms



photonAnalysis = cms.EDAnalyzer("PhotonAnalyzer",
    ComponentName = cms.string('photonAnalysis'),
    phoProducer = cms.InputTag('gedPhotons'),
    analyzerName = cms.string('gedPhotonAnalyzer'),
    photonIDLoose = cms.InputTag('PhotonIDProd:PhotonCutBasedIDLoose'),
    photonIDTight = cms.InputTag('PhotonIDProd:PhotonCutBasedIDTight'),																

    barrelRecHitProducer = cms.InputTag('reducedEcalRecHitsEB'),								
    endcapRecHitProducer = cms.InputTag('reducedEcalRecHitsEE'),

    triggerEvent = cms.InputTag("hltTriggerSummaryAOD",""),                            
    prescaleFactor = cms.untracked.int32(1),

    useBinning = cms.bool(False),
    useTriggerFiltering = cms.bool(False),                             
    standAlone = cms.bool(False),
                                
    minPhoEtCut = cms.double(20.0),
    maxPhoEta   = cms.double(2.5),
    invMassEtCut = cms.double(20.0),                             
              
    cutStep = cms.double(20.0),
    numberOfSteps = cms.int32(1),
    minimalSetOfHistos = cms.bool(True),
    excludeBkgHistos = cms.bool(True),                            

    isHeavyIon = cms.untracked.bool(False),

    # DBE verbosity
    Verbosity = cms.untracked.int32(0),
                                # 1 provides basic output
                                # 2 provides output of the fill step + 1
                                # 3 provides output of the store step + 2
                                
    isolationStrength = cms.int32(0),
                                # 0 => Loose Photon = GoodCandidate
                                ### CAREFULL for the time being only Loos exists
                                # 1 => Tight Photon = GoodCandidate



    ##### Histogram Ranges and Bins                               

    eBin = cms.int32(500),
    eMin = cms.double(0.0),
    eMax = cms.double(500.0),
                                
    etBin = cms.int32(500),
    etMin = cms.double(0.0),
    etMax = cms.double(500.0),

    sumBin = cms.int32(230),
    sumMin = cms.double(-3.0),
    sumMax = cms.double(20.0),

    etaBin = cms.int32(200),                               
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),

    phiBin = cms.int32(200),
    phiMin = cms.double(-3.14),
    phiMax = cms.double(3.14),
    
    r9Bin = cms.int32(120),
    r9Min = cms.double(0.0),
    r9Max = cms.double(1.2),

    hOverEBin = cms.int32(200),
    hOverEMin = cms.double(0),                               
    hOverEMax = cms.double(0.5), 

    numberBin = cms.int32(10),
    numberMin = cms.double(0),                               
    numberMax = cms.double(10), 

    xBin = cms.int32(300),
    xMin = cms.double(-60),                               
    xMax = cms.double(60),
                                
    yBin = cms.int32(300),
    yMin = cms.double(-60),                               
    yMax = cms.double(60),
                                
    rBin = cms.int32(400),
    rMin = cms.double(0),                               
    rMax = cms.double(80),                                

    zBin = cms.int32(400),
    zMin = cms.double(-200),                               
    zMax = cms.double(200),

    dEtaTracksBin = cms.int32(100),
    dEtaTracksMin = cms.double(-0.2),
    dEtaTracksMax = cms.double(0.2),

    dPhiTracksBin = cms.int32(100),
    dPhiTracksMin = cms.double(-0.5),
    dPhiTracksMax = cms.double(0.5),

    sigmaIetaBin = cms.int32(200),
    sigmaIetaMin = cms.double(0.0),
    sigmaIetaMax = cms.double(0.05),

    eOverPBin = cms.int32(100),
    eOverPMin = cms.double(0.0),
    eOverPMax = cms.double(5.0),

    chi2Bin = cms.int32(100),
    chi2Min = cms.double(0.0),
    chi2Max = cms.double(20.0),                                
                                

                                
    OutputFileName = cms.string('DQMOfflinePhotonsAfterFirstStep.root'),


)
