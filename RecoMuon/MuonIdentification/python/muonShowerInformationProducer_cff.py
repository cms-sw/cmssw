from RecoMuon.MuonIdentification.muonShowerInformation_cfi import *
muonShowerInformation = cms.EDProducer("MuonShowerInformationProducer",
                                           MuonServiceProxy,
    muonCollection = cms.InputTag("muons"),
    trackCollection = cms.InputTag("generalTracks"),
    ShowerInformationFillerParameters = MuonShowerParameters
)
