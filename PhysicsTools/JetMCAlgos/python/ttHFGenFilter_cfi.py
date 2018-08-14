import FWCore.ParameterSet.Config as cms

ttHFGenFilter = cms.EDFilter("ttHFGenFilter",

    genParticles = cms.InputTag("genParticles"),
    genBHadFlavour = cms.InputTag("matchGenBHadron", "genBHadFlavour"),
    genBHadFromTopWeakDecay = cms.InputTag("matchGenBHadron", "genBHadFromTopWeakDecay"),
    genBHadPlusMothers = cms.InputTag("matchGenBHadron", "genBHadPlusMothers"),
    genBHadPlusMothersIndices = cms.InputTag("matchGenBHadron", "genBHadPlusMothersIndices"),
    genBHadIndex = cms.InputTag("matchGenBHadron", "genBHadIndex"),
    OnlyHardProcessBHadrons = cms.bool(False),
    taggingMode               = cms.bool(False)

)
