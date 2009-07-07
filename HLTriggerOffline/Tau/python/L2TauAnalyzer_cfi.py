import FWCore.ParameterSet.Config as cms

L2TauAnalyzer = cms.EDAnalyzer("L2TauAnalyzer",
    GenJetCollection = cms.InputTag("iterativeCone5GenJetsNoNuBSM"),
    IsSignal = cms.bool(True),
    MatchedCollection = cms.InputTag("TauMcInfoProducer","Jets"),
    outputFileName = cms.string('output_L2.root'),
    L2InfoAssociationInput = cms.InputTag("doubleTauL2Producer","L2TauIsolationInfoAssociator")
)


