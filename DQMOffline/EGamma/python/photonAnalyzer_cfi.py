import FWCore.ParameterSet.Config as cms

PhotonAnalysis = cms.EDAnalyzer("PhotonAnalyzer",

    Name = cms.untracked.string('PhotonAnalysis'),

    phoProducer = cms.string('photons'),
    photonCollection = cms.string(''),


    cutStep = cms.double(50.0),
    numberOfSteps = cms.int32(2),                          

    useBinning = cms.bool(False),
                             

    minPhoEtCut = cms.double(0.0),
              

    # DBE verbosity
    Verbosity = cms.untracked.int32(0),
                                # 1 provides basic output
                                # 2 provides output of the fill step + 1
                                # 3 provides output of the store step + 2
                                
    isolationStrength = cms.int32(2),
                                # 1 => Loose EM
                                # 2 => Loose Photon
                                # 3 => Tight Photon

 

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
    OutputFileName = cms.string('/afs/cern.ch/user/l/lantonel/scratch0/CMSSW_2_1_4/src/DQMOffline/EGamma/PhotonsTest.root'),

)


