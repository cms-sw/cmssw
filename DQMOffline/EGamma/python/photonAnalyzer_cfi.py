import FWCore.ParameterSet.Config as cms

photonAnalysis = cms.EDAnalyzer("PhotonAnalyzer",

    Name = cms.untracked.string('photonAnalysis'),

    phoProducer = cms.string('photons'),
    photonCollection = cms.string(''),

    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),

    prescaleFactor = cms.untracked.int32(1),

    cutStep = cms.double(50.0),
    numberOfSteps = cms.int32(2),                          

    useBinning = cms.bool(False),
                             

    minPhoEtCut = cms.double(0.0),
              

    # DBE verbosity
    Verbosity = cms.untracked.int32(0),
                                # 1 provides basic output
                                # 2 provides output of the fill step + 1
                                # 3 provides output of the store step + 2
                                
    isolationStrength = cms.int32(1),
                                # 1 => Loose Photon
                                # 2 => Tight Photon
                                

 

    eBin = cms.int32(250),
    eMin = cms.double(0.0),
    eMax = cms.double(250.0),
                                
    etBin = cms.int32(200),
    etMin = cms.double(0.0),
    etMax = cms.double(200.0),
                                
    etaBin = cms.int32(100),                               
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),

    phiBin = cms.int32(100),                               
    phiMin = cms.double(-3.14),
    phiMax = cms.double(3.14),
                                
    r9Bin = cms.int32(100),
    r9Min = cms.double(0.0),
    r9Max = cms.double(1.1),

    dEtaTracksBin = cms.int32(100),
    dEtaTracksMin = cms.double(-0.2),
    dEtaTracksMax = cms.double(0.2),

    dPhiTracksBin = cms.int32(100),
    dPhiTracksMin = cms.double(-0.5),
    dPhiTracksMax = cms.double(0.5),
# parameters for pizer finding                                
    seleXtalMinEnergy = cms.double(0.0),
    clusSeedThr = cms.double(0.5),
    clusPhiSize = cms.int32(3),
    clusEtaSize = cms.int32(3),
    ParameterLogWeighted = cms.bool(True),                          
    ParameterX0 = cms.double(0.89),
    ParameterW0 = cms.double(4.2),
    ParameterT0_barl = cms.double(5.7),
    selePtGammaOne = cms.double(0.9),
    selePtGammaTwo = cms.double(0.9),                          
    seleS4S9GammaOne = cms.double(0.85),
    seleS4S9GammaTwo = cms.double(0.85),
    selePtPi0 = cms.double(2.5),
    selePi0Iso = cms.double(0.5),
    selePi0BeltDR = cms.double(0.2),
    selePi0BeltDeta = cms.double(0.05),
    seleMinvMaxPi0 = cms.double(0.5),
    seleMinvMinPi0 = cms.double(0.0),
#                                
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('DQMOfflinePhotons.root'),

)


