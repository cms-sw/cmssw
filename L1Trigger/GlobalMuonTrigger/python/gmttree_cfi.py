import FWCore.ParameterSet.Config as cms

gmttree = cms.EDAnalyzer("L1MuGMTTree",
    GeneratorInputTag = cms.InputTag("none"),
    SimulationInputTag = cms.InputTag("none"),
    GTInputTag = cms.InputTag("none"),
    GTEvmInputTag = cms.InputTag("none"),
    GMTInputTag = cms.InputTag("l1GmtEmulDigis"),

    PhysVal = cms.bool(True),
    OutputFile = cms.untracked.string('gmttree.root')
)



