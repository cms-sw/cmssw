from RecoMuon.MuonIdentification.muonShowerInformation_cfi import *
muonShowerInformation = cms.EDProducer("MuonShowerInformationProducer",
                                           MuonServiceProxy,
    muonCollection = cms.InputTag("muons1stStep"),
    trackCollection = cms.InputTag("generalTracks"),
    ShowerInformationFillerParameters = MuonShowerParameters.MuonShowerInformationFillerParameters
)
