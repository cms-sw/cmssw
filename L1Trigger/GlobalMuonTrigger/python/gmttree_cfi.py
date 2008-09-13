import FWCore.ParameterSet.Config as cms

gmttree = cms.EDFilter("L1MuGMTTree",
    OutputFile = cms.untracked.string('gmttree.root'),
    GeneratorInputTag = cms.InputTag("none"),
    PhysVal = cms.bool(True),
    GTEvmInputTag = cms.InputTag("none"),
    GMTInputTag = cms.InputTag("l1GmtEmulDigis"),
    GTInputTag = cms.InputTag("none"),
    SimulationInputTag = cms.InputTag("none"),
)



