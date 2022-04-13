import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.PatPhotonDRNCorrectionProducer_cfi import PatPhotonDRNCorrectionProducer

patPhotonsDRN = cms.EDProducer("PatPhotonDRNCorrectionProducer",
    particleSource = cms.InputTag("selectedPatPhotons"),
    rhoName = cms.InputTag("fixedGridRhoFastjetAll"),

    Client = cms.PSet(
      mode = cms.string("Async"),
      modelName = cms.string("photonObjectEnsemble"),
      modelConfigPath = cms.FileInPath("RecoEgamma/EgammaPhotonProducers/data/models/photonObjectEnsemble/config.pbtxt"),
      allowedTries = cms.untracked.uint32(1),
      timeout = cms.untracked.uint32(10),
      useSharedMemory = cms.untracked.bool(True),
    )
)
