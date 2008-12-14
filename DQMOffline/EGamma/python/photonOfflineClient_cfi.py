import FWCore.ParameterSet.Config as cms

photonOfflineClient = cms.EDAnalyzer("PhotonOfflineClient",

    Name = cms.untracked.string('photonOfflineClient'),

    cutStep = cms.double(50.0),
    numberOfSteps = cms.int32(2),

    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('DQMOfflinePhotonsSecondStep.root'),
)
