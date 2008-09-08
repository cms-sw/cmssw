import FWCore.ParameterSet.Config as cms

PhotonAnalysis = cms.EDAnalyzer("PhotonAnalyzer",

    Name = cms.untracked.string('PhotonAnalysis'),

    phoProducer = cms.string('photons'),
    photonCollection = cms.string(''),

 #   hbheModule = cms.string('hbhereco'),
 #   hbheInstance = cms.string(''),
                                
 #   trackProducer = cms.InputTag("generalTracks"),

 ## for 2_1_4
#    bcBarrelProducer = cms.string("hybridSuperClusters"),
#    bcEndcapProducer = cms.string("multi5x5BasicClusters"),                                       
#    bcBarrelCollection = cms.string("hybridBarrelBasicClusters"),                            
#    bcEndcapCollection = cms.string("multi5x5EndcapBasicClusters"),

 #   scBarrelProducer = cms.InputTag('correctedHybridSuperClusters'),
 #   scEndcapProducer = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'),

 #   barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
 #   endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),




    cutStep = cms.double(32.0),
    numberOfSteps = cms.int32(2),                          


    minPhoEtCut = cms.double(0.0),
    minBcEtCut = cms.double(0.0),
    minTrackPtCut = cms.double(1.5),
    minHcalHitEtCut = cms.double(0.0),

    lipCut = cms.double(2.0),                            

    ecalEtaStrip = cms.double(0.02),                  

    # DBE verbosity
    Verbosity = cms.untracked.int32(0),
                                # 1 provides basic output
                                # 2 provides output of the fill step + 1
                                # 3 provides output of the store step + 2
                                

 

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
                                

    OutputMEsInRootFile = cms.bool(True),
    OutputFileName = cms.string('/afs/cern.ch/user/l/lantonel/scratch0/CMSSW_2_1_4/src/DQMOffline/EGamma/TedsPhotonsTest.root'),

)


