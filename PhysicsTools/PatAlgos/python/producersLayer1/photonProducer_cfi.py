import FWCore.ParameterSet.Config as cms

allLayer1Photons = cms.EDProducer("PATPhotonProducer",
    isolation = cms.PSet(
        hcal = cms.PSet(
            src = cms.InputTag("layer0PhotonIsolations","egammaPhotonTowersDeposits"),
            deltaR = cms.double(0.3)
        ),
        tracker = cms.PSet(
            threshold = cms.double(1.5),
            src = cms.InputTag("layer0PhotonIsolations","egammaPhotonTkDeposits"),
            deltaR = cms.double(0.3)
        ),
        ecal = cms.PSet(
            src = cms.InputTag("layer0PhotonIsolations","egammaPhotonEcalDeposits"),
            deltaR = cms.double(0.3)
        )
    ),
    embedSuperCluster = cms.bool(False),
    photonSource = cms.InputTag("allLayer0Photons"),
    isoDeposits = cms.PSet(
        hcal = cms.InputTag("layer0PhotonIsolations","egammaPhotonTowersDeposits"),
        tracker = cms.InputTag("layer0PhotonIsolations","egammaPhotonTkDeposits"),
        ecal = cms.InputTag("layer0PhotonIsolations","egammaPhotonEcalDeposits")
    )
)


