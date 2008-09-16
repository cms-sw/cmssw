import FWCore.ParameterSet.Config as cms

l1PromptAnalysis = cms.EDFilter("L1PromptAnalysis",
    PhysVal = cms.bool(True),
    OutputFile = cms.untracked.string('gmttree.root'),
    GeneratorInputTag = cms.InputTag("none"),
    SimulationInputTag = cms.InputTag("none"),
    GMTInputTag = cms.InputTag("l1GtUnpack"),
    GTEvmInputTag = cms.InputTag("l1GtEvmUnpack","","L1Prompt"),
    GTInputTag = cms.InputTag("l1GtUnpack","","L1Prompt"),
)



