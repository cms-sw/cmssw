import FWCore.ParameterSet.Config as cms

photonAnalysis = cms.EDAnalyzer("PhotonAnalyzer",
    OutputMEsInRootFile = cms.bool(True),
    dEtaTracksMin = cms.double(-0.2),
    eMin = cms.double(0.0),
    hcalIsolExtR = cms.double(0.3),
    dEtaTracksMax = cms.double(0.2),
    hcalIsolInnR = cms.double(0.0),
    scEndcapProducer = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'),
    minTrackPtCut = cms.double(1.5),
    minBcEtCut = cms.double(0.0),
    bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    etBin = cms.int32(100),
    etaMax = cms.double(2.5),
    phiMax = cms.double(3.14),
    r9Max = cms.double(1.1),
    phiMin = cms.double(-3.14),
    lipCut = cms.double(2.0),
    eMax = cms.double(100.0),
    bcEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    OutputFileName = cms.string('DQMPhotonHisto.root'),
    photonCollection = cms.string(''),
    etaBin = cms.int32(100),
    dEtaTracksBin = cms.int32(100),
    hbheInstance = cms.string(''),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    dPhiTracksMax = cms.double(0.5),
    # string  bcProducer  =        "islandBasicClusters"
    # string  bcBarrelCollection = "islandBarrelBasicClusters"
    # string  bcEndcapCollection = "islandEndcapBasicClusters"
    hbheModule = cms.string('hbhereco'),
    trackProducer = cms.InputTag("generalTracks"),
    # DBE verbosity
    Verbosity = cms.untracked.int32(0),
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    trkIsolInnR = cms.double(0.03),
    r9Min = cms.double(0.0),
    ecalIsolR = cms.double(0.35),
    dPhiTracksMin = cms.double(-0.5),
    phiBin = cms.int32(100),
    r9Bin = cms.int32(100),
    etMin = cms.double(0.0),
    trkIsolExtR = cms.double(0.3),
    eBin = cms.int32(100),
    maxNumOfTracksInCone = cms.int32(3),
    etMax = cms.double(100.0),
    hcalEtSumCut = cms.double(6.0),
    phoProducer = cms.string('photons'),
    minHcalHitEtCut = cms.double(0.0),
    etaMin = cms.double(-2.5),
    minPhoEtCut = cms.double(5.0),
    dPhiTracksBin = cms.int32(100),
    scBarrelProducer = cms.InputTag('correctedHybridSuperClusters'),
    Name = cms.untracked.string('photonAnalysis'),
    trkPtSumCut = cms.double(9999.0),
    ecalEtSumCut = cms.double(5.0),
    ecalEtaStrip = cms.double(0.02)                             
)


