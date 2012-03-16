
import FWCore.ParameterSet.Config as cms

process = cms.Process("READCONVERTED")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
       'file:converted.root'
   )
)

process.eca= cms.EDAnalyzer("EventContentAnalyzer",
   getData = cms.untracked.bool(True),
   listContent = cms.untracked.bool(False)
)

process.p = cms.Path(process.eca)

