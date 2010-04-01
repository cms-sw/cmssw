import FWCore.ParameterSet.Config as cms

# dummy corrector needed to set record's IOV
DummyCorrector = cms.ESSource("SimpleJetCorrectionService",
    scale = cms.double(1.0),
    label = cms.string('DummyCorrector')
)

es_prefer_DummyCorrector = cms.ESPrefer("SimpleJetCorrectionService","DummyCorrector")

