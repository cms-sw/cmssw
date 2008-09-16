import FWCore.ParameterSet.Config as cms

UEAnalysisIconeJetParameters = cms.PSet(
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(5.0),
    inputEtMin = cms.double(0.9),
    coneRadius = cms.double(0.5),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('GenJet'),
    inputEMin = cms.double(0.0)
)
IC5GenJet = cms.EDProducer("IterativeConeJetProducer",
    UEAnalysisIconeJetParameters,
    src = cms.InputTag("goodParticles")
)

IC5ChgGenJet = cms.EDProducer("IterativeConeJetProducer",
    UEAnalysisIconeJetParameters,
    src = cms.InputTag("chargeParticles")
)

IC5TracksJet = cms.EDProducer("IterativeConeJetProducer",
    UEAnalysisIconeJetParameters,
    src = cms.InputTag("goodTracks")
)

IC5GenJet500 = cms.EDProducer("IterativeConeJetProducer",
    UEAnalysisIconeJetParameters,
    src = cms.InputTag("goodParticles")
)

IC5ChgGenJet500 = cms.EDProducer("IterativeConeJetProducer",
    UEAnalysisIconeJetParameters,
    src = cms.InputTag("chargeParticles")
)

IC5TracksJet500 = cms.EDProducer("IterativeConeJetProducer",
    UEAnalysisIconeJetParameters,
    src = cms.InputTag("goodTracks")
)

UEAnalysisJetsOnlyMC = cms.Sequence(IC5GenJet*IC5ChgGenJet*IC5GenJet500*IC5ChgGenJet500)
UEAnalysisJetsOnlyReco = cms.Sequence(IC5TracksJet*IC5TracksJet500)
UEAnalysisJets = cms.Sequence(UEAnalysisJetsOnlyMC*UEAnalysisJetsOnlyReco)
IC5TracksJet.jetType = 'BasicJet'
IC5GenJet500.inputEtMin = 0.5
IC5ChgGenJet500.inputEtMin = 0.5
IC5TracksJet500.jetType = 'BasicJet'
IC5TracksJet500.inputEtMin = 0.5


