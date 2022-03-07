import FWCore.ParameterSet.Config as cms

patPhotonsDRN = cms.EDProducer("PatPhotonDRNCorrectionProducer",
    particleSource = cms.InputTag("selectedPatPhotons"),
    rhoName = cms.InputTag("fixedGridRhoFastjetAll"),
    reducedEcalRecHitsEB = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEcalRecHitsEE = cms.InputTag("reducedEcalRecHitsEE"),
    reducedEcalRecHitsES = cms.InputTag("reducedEcalRecHitsES"),

    Client = cms.PSet(
      mode = cms.string("Async"),
      modelName = cms.string("photonObjectEnsemble"),
      modelConfigPath = cms.FileInPath("RecoEgamma/EgammaPhotonProducers/data/models/photonObjectEnsemble/config.pbtxt"),
      allowedTries = cms.untracked.uint32(1),
      timeout = cms.untracked.uint32(10),
      useSharedMemory = cms.untracked.bool(False),
      verbose = cms.untracked.bool(True)
    )
)
