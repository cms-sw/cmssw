import FWCore.ParameterSet.Config as cms
process = cms.Process("mytest1")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

process.load("Alignment.TrackerAlignment.TkAlCaSkimTreeMerger_cff")
process.AlignmentTreeMerger.FileList="<DQMLIST>"
process.AlignmentTreeMerger.TreeName='AlignmentHitMap'
process.AlignmentTreeMerger.OutputFile="<DQMTOTFILE>"
process.AlignmentTreeMerger.NhitsMaxLimit=400

process.path = cms.Path(process.AlignmentTreeMerger)
