import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("ANA1")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:"+sys.argv[-1])
)

process.analyzer = cms.EDAnalyzer("IntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer"),
    valueMustMatch = cms.untracked.int32(1)
)

process.p = cms.Path(process.analyzer)
