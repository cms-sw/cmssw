import FWCore.ParameterSet.Config as cms

tauJetSelectorForHLTTrackSeeding = cms.EDProducer(
    "TauJetSelectorForHLTTrackSeeding",
    inputTrackJetTag = cms.InputTag( "hltAntiKT5TrackJetsIter0" ),
    inputCaloJetTag = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
    inputTrackTag = cms.InputTag( "hltPFlowTrackSelectionHighPurity"),
    ptMinCaloJet = cms.double(5.0),
    etaMinCaloJet = cms.double(-2.7),
    etaMaxCaloJet = cms.double(+2.7),
    tauConeSize = cms.double(0.2),
    isolationConeSize = cms.double(0.5),
    fractionMinCaloInTauCone = cms.double(0.8),
    fractionMaxChargedPUInCaloCone = cms.double(0.2),
    ptTrkMaxInCaloCone = cms.double(1.0),
    nTrkMaxInCaloCone = cms.int32(0)
)
