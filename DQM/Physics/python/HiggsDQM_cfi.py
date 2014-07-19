import FWCore.ParameterSet.Config as cms

HiggsDQM = cms.EDAnalyzer("HiggsDQM",
    elecTriggerPathToPass    = cms.string("HLT_Ele10_LW_L1R"),
    muonTriggerPathToPass    = cms.string("HLT_Mu9"),
    triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),
    muonCollection           = cms.InputTag("muons"),
    electronCollection       = cms.InputTag("gedGsfElectrons"),
    caloJetCollection        = cms.InputTag("ak4CaloJets"),
    caloMETCollection        = cms.InputTag("caloMetM"),
    pfMETCollection          = cms.InputTag("pfMet"),
    genParticleCollection    = cms.InputTag("genParticles"),

    PtThrMu1 = cms.untracked.double(3.0),
    PtThrMu2 = cms.untracked.double(3.0)
)

