import FWCore.ParameterSet.Config as cms

hltInclusiveVBFSource = cms.EDAnalyzer(
    "HLTInclusiveVBFSource",
    dirname     = cms.untracked.string("HLT/InclusiveVBF"),
    processname = cms.string("HLT"),
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    #
    debug = cms.untracked.bool(False),
    #
    CaloJetCollectionLabel = cms.InputTag("ak4CaloJets"),
    CaloMETCollectionLabel = cms.InputTag("met"),
    PFJetCollectionLabel = cms.InputTag("ak4PFJets"),
    PFMETCollectionLabel = cms.InputTag("pfMet"),
    #
    minPtHigh    = cms.untracked.double(40.),
    minPtLow     = cms.untracked.double(40.),
    minDeltaEta  = cms.untracked.double(3.5),
    minInvMass   = cms.untracked.double(1000.),
    deltaRMatch  = cms.untracked.double(0.1),
    etaOpposite  = cms.untracked.bool(True)
)

