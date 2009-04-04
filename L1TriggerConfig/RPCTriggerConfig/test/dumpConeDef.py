import FWCore.ParameterSet.Config as cms

process = cms.Process("dump")


process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeDefinition_cff")


process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.write = cms.EDAnalyzer("DumpConeDefinition",
          fileName = cms.string("coneDefinitionDump.txt"),
)


process.p1 = cms.Path(process.write)
