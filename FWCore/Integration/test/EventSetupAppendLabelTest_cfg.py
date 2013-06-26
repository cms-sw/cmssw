import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource")

process.WhatsItAnalyzer = cms.EDAnalyzer("WhatsItAnalyzer",
    expectedValues = cms.untracked.vint32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer",
    doodadLabel = cms.string('Two')
)

# This one is here only to test that we are allowed to
# add the special appendToDataLabel parameter without
# putting it in the fillDescriptions function of the ESProducer.
process.WhatsItESProducer1 = cms.ESProducer("WhatsItESProducer",
    doodadLabel = cms.string('Two'),
    appendToDataLabel = cms.string('foo')
)

process.DoodadESSource = cms.ESSource("DoodadESSource",
    appendToDataLabel = cms.string('Two')
)

process.p = cms.Path(process.WhatsItAnalyzer)
