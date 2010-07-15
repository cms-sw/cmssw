import FWCore.ParameterSet.Config as cms

process = cms.Process("TREESPLITTER")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

process.TreeSplitterModule = cms.EDAnalyzer(
    "TreeSplitter",

    InputFileName = cms.string("/home/demattia/MuScleFit/Data/Zmumu/Zmumu.root"),
    OutputFileName = cms.string("subSample.root"),
    MaxEvents = cms.int32(-1),
    SubSampleFirstEvent = cms.uint32(0),
    SubSampleMaxEvents = cms.uint32(1000)
)

process.p1 = cms.Path(process.TreeSplitterModule)

