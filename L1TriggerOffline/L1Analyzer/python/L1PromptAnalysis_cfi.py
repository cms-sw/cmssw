import FWCore.ParameterSet.Config as cms

l1PromptAnalysis = cms.EDAnalyzer("L1PromptAnalysis",
    verbose = cms.untracked.bool(False),
    PhysVal = cms.bool(True),
    OutputFile = cms.untracked.string('gmttree.root'),
    GeneratorInputTag = cms.InputTag("none"),
    SimulationInputTag = cms.InputTag("none"),
    GMTInputTag = cms.InputTag("l1GtUnpack"),
    GTEvmInputTag = cms.InputTag("l1GtEvmUnpack","","L1Prompt"),
    GTInputTag = cms.InputTag("l1GtUnpack","","L1Prompt"),
    gctCentralJetsSource = cms.InputTag("l1GctHwDigis","cenJets","L1Prompt"),
    gctNonIsoEmSource = cms.InputTag("l1GctHwDigis","nonIsoEm","L1Prompt"),
    gctForwardJetsSource = cms.InputTag("l1GctHwDigis","forJets","L1Prompt"),
    gctIsoEmSource = cms.InputTag("l1GctHwDigis","isoEm","L1Prompt"),
    gctEnergySumsSource = cms.InputTag("l1GctHwDigis","","L1Prompt"),
    gctTauJetsSource = cms.InputTag("l1GctHwDigis","tauJets","L1Prompt"),
    rctSource = cms.InputTag("l1GctHwDigis","","L1Prompt"),
    dttfSource = cms.InputTag("l1dttfunpack","","L1Prompt")
    )



