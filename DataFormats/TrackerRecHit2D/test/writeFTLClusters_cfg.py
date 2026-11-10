import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9

options = VarParsing("analysis")
options.outputFile = "fakeFTLClusters.root"
options.parseArguments()

process = cms.Process("FAKEFTL", Phase2C26I13M9)
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.Geometry.GeometryExtendedRun4D127Reco_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase2_realistic_T35", "")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))
process.mtdClusters = cms.EDProducer("TestWriteFTLClusters")
process.p = cms.Path(process.mtdClusters)
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName=cms.untracked.string(options.outputFile),
    outputCommands=cms.untracked.vstring("drop *", "keep *_mtdClusters_*_*"),
)
process.e = cms.EndPath(process.out)
