import FWCore.ParameterSet.Config as cms

diJetAnalyzer = cms.EDAnalyzer(
    'DiJetAnalyzer',
    pfJetCollName       = cms.string('DiJetsProd:ak4PFJetsCHS'),
    pfJetCorrName       = cms.string('ak4PFCHSL1FastL2L3'),
    hbheRecHitName      = cms.string('DiJetsProd:hbhereco'),
    hfRecHitName        = cms.string('DiJetsProd:hfreco'),
    hoRecHitName        = cms.string('DiJetsProd:horeco'),
    pvCollName          = cms.string('DiJetsProd:offlinePrimaryVertices'),
    rootHistFilename    = cms.string('dijettree.root'),
    maxDeltaEta         = cms.double(1.5),
    minTagJetEta        = cms.double(0.0),
    maxTagJetEta        = cms.double(5.0),
    minSumJetEt         = cms.double(50.),
    minJetEt            = cms.double(20.),
    maxThirdJetEt       = cms.double(75.),
    debug               = cms.untracked.bool(False)
    )
