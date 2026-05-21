import FWCore.ParameterSet.Config as cms

HiGenParticleAna = cms.EDAnalyzer(
    'HiGenAnalyzer',
    doVertex = cms.untracked.bool(False),
    etaMax = cms.untracked.double(2.5),
    ptMin = cms.untracked.double(5),
    chargedOnly = cms.untracked.bool(False),
    stableOnly = cms.untracked.bool(True),
    src = cms.untracked.InputTag("generator"),
    genParticleSrc = cms.InputTag("packedGenParticles"),
    signalGenParticleSrc = cms.InputTag("packedGenParticlesSignal"),
    genHIsrc = cms.untracked.InputTag("heavyIon"),
    doParticles = cms.untracked.bool(True),
    doHI = cms.untracked.bool(False)  ## Relevant info (its AOD counterpart is edm::GenHIEvent "heavyIon") is missing currently.
    )
