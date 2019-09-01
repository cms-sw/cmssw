import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("ANA1")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:"+sys.argv[-1])
)

process.analyzer = cms.EDAnalyzer("SwitchProducerProvenanceAnalyzer",
    src1 = cms.InputTag("intProducer"),
    src2 = cms.InputTag("intProducer", "other"),
    producerPrefix = cms.string("intProducer"),
    aliasMode = cms.bool(False)
)

process.analyzerAlias = cms.EDAnalyzer("SwitchProducerProvenanceAnalyzer",
    src1 = cms.InputTag("intProducerAlias"),
    src2 = cms.InputTag("intProducerAlias", "other"),
    producerPrefix = cms.string("intProducer"),
    aliasMode = cms.bool(True)
)

process.p = cms.Path(process.analyzer + process.analyzerAlias)
