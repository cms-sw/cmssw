import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource",
  numberEventsInRun = cms.untracked.uint32(3)
)

process.WhatsItAnalyzer = cms.EDAnalyzer("WhatsItAnalyzer",
    expectedValues = cms.untracked.vint32(0,0,0,1,1))

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.p = cms.Path(process.WhatsItAnalyzer)

process.options = cms.untracked.PSet(forceEventSetupCacheClearOnNewRun = cms.untracked.bool(True))