import FWCore.ParameterSet.Config as cms

deepMETSonicProducer = cms.EDProducer("DeepMETSonicProducer",
    Client = cms.PSet(
        timeout = cms.untracked.uint32(300),
        mode = cms.string("Async"),
        modelName = cms.string("deepmet"),
        modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/deepmet/config.pbtxt"),
        # version "1" is the resolutionTune
        # version "2" is the responeTune
        modelVersion = cms.string("1"),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(0),
        useSharedMemory = cms.untracked.bool(True),
        compression = cms.untracked.string(""),
    ),
    pf_src = cms.InputTag("packedPFCandidates"),
)
