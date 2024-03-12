from RecoMuon.MuonIdentification.muonShowerInformation_cfi import *
muonShowerInformation = cms.EDProducer("MuonShowerInformationProducer",
                                           MuonServiceProxy,
    muonCollection = cms.InputTag("muons1stStep"),
    trackCollection = cms.InputTag("generalTracks"),
    ShowerInformationFillerParameters = MuonShowerParameters.MuonShowerInformationFillerParameters
)
# foo bar baz
# Aee8d8p7iR4Wo
# xxv8RcCvXwv99
