import FWCore.ParameterSet.Config as cms

process = cms.Process("TREESPLITTER")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

process.TreeSplitterModule = cms.EDAnalyzer(
    "TreeSplitter",

    InputFileName = cms.string("tree_Summer10JPsiPromptSTARTUP.root"),
    OutputFileName = cms.string("SubSample.root"),
    MaxEvents = cms.int32(MAXEVENTS),
    SubSampleFirstEvent = cms.uint32(SUBSAMPLEFIRSTEVENT),
    SubSampleMaxEvents = cms.uint32(SUBSAMPLEMAXEVENTS)
)

process.p1 = cms.Path(process.TreeSplitterModule)

