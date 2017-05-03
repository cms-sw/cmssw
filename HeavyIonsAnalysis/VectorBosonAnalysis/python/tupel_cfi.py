import FWCore.ParameterSet.Config as cms

tupel = cms.EDAnalyzer(
    "Tupel",
    trigger      = cms.InputTag( "patTrigger" ),
    #  triggerEvent = cms.InputTag( "patTriggerEvent" ),
    #  triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    photonSrc    = cms.untracked.InputTag("patPhotons"),
    vtxSrc       = cms.untracked.InputTag("offlinePrimaryVertices"),
    electronSrc  = cms.untracked.InputTag("patElectrons"),
    muonSrc      = cms.untracked.InputTag("patMuonsWithTrigger"),
    #  tauSrc      = cms.untracked.InputTag("slimmedPatTaus"),
    jetSrc       = cms.untracked.InputTag("akPu4PFpatJetsWithBtagging"),
    metSrc       = cms.untracked.InputTag("patMETsPF"),
    genSrc       = cms.untracked.InputTag("genParticles"),
    gjetSrc      = cms.untracked.InputTag('ak4HiGenJets'),
    muonMatch    = cms.string( 'muonTriggerMatchHLTMuons' ),
    muonMatch2   = cms.string( 'muonTriggerMatchHLTMuons2' ),
    elecMatch    = cms.string( 'elecTriggerMatchHLTElecs' ),
    #  mSrcRho      = cms.untracked.InputTag('fixedGridRhoFastjetAll'),
    CalojetLabel = cms.untracked.InputTag('ak4CalopatJets'),
    metSource    = cms.VInputTag("slimmedMETs","slimmedMETs","slimmedMETs","slimmedMETs"),
    lheSource    = cms.untracked.InputTag('source')
)
