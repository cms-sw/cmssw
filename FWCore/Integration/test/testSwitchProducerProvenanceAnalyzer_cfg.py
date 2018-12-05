import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("ANA1")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:"+sys.argv[-1])
)

process.analyzer = cms.EDAnalyzer("SwitchProducerProvenanceAnalyzer",
    src1 = cms.InputTag("intProducer"),
    src2 = cms.InputTag("intProducer", "other")
)

process.p = cms.Path(process.analyzer)
