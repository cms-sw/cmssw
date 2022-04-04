import FWCore.ParameterSet.Config as cms

patElectronsDRN = cms.EDProducer("PatElectronDRNCorrectionProducer",
    particleSource = cms.InputTag("selectedPatElectrons"),
    rhoName = cms.InputTag("fixedGridRhoFastjetAll"),

    Client = cms.PSet(
      mode = cms.string("Async"),
      modelName = cms.string("electronObjectEnsemble"),
      modelConfigPath = cms.FileInPath("RecoEgamma/EgammaElectronProducers/data/models/electronObjectEnsemble/config.pbtxt"),
      allowedTries = cms.untracked.uint32(1),
      timeout = cms.untracked.uint32(10),
      useSharedMemory = cms.untracked.bool(False),
    )
)
