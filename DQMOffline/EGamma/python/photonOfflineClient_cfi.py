import FWCore.ParameterSet.Config as cms

photonOfflineClient = cms.EDAnalyzer("PhotonOfflineClient",

    Name = cms.untracked.string('photonOfflineClient'),


    standAlone = cms.bool(False),
    batch = cms.bool(False),                                     

    cutStep = cms.double(50.0),
    numberOfSteps = cms.int32(2),

    #etBin = cms.int32(300),
    etBin = cms.int32(600),
    etMin = cms.double(0.0),
    etMax = cms.double(3000.0),
                                 
    #etaBin = cms.int32(100),                               
    etaBin = cms.int32(200),                               
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
                                     
    InputFileName = cms.untracked.string("DQMOfflinePhotonsBatch.root"),
                                     
    OutputFileName = cms.string('DQMOfflinePhotonsStandAlone.root'),
)
