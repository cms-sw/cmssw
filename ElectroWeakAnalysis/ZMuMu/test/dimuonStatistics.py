import FWCore.ParameterSet.Config as cms

process = cms.Process("dimuonStatistics")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "file:/scratch1/cms/data/summer08/skim/dimuons_skim_zmumu.root"
   )
)

process.stat = cms.EDAnalyzer(
    "DimuonStatistics",
    src = cms.InputTag("dimuonsGlobal"),
    )

process.path = cms.Path(
    process.stat
    )



