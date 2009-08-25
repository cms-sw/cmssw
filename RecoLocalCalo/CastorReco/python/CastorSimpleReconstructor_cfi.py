import FWCore.ParameterSet.Config as cms

castorreco = cms.EDFilter("CastorSimpleReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("simCastorDigis"),
    samplesToAdd = cms.int32(1),
    Subdetector = cms.string('CASTOR'),
    firstSample = cms.int32(3),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False)
)
