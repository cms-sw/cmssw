import FWCore.ParameterSet.Config as cms

# services providing different jet flavor corrections
L5FlavorJetCorrectorGluon = cms.ESSource("L5FlavorCorrectionService",
    section = cms.string('g'),
    tagName = cms.string('L5Flavor_v1'),
    label = cms.string('L5FlavorJetCorrectorGluon')
)

L5FlavorJetCorrectorUds = cms.ESSource("L5FlavorCorrectionService",
    section = cms.string('uds'),
    tagName = cms.string('L5Flavor_v1'),
    label = cms.string('L5FlavorJetCorrectorUds')
)

L5FlavorJetCorrectorC = cms.ESSource("L5FlavorCorrectionService",
    section = cms.string('c'),
    tagName = cms.string('L5Flavor_v1'),
    label = cms.string('L5FlavorJetCorrectorC')
)

L5FlavorJetCorrectorB = cms.ESSource("L5FlavorCorrectionService",
    section = cms.string('b'),
    tagName = cms.string('L5Flavor_v1'),
    label = cms.string('L5FlavorJetCorrectorB')
)


