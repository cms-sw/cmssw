import FWCore.ParameterSet.Config as cms

# impactParameterMVAComputer btag computer
impactParameterMVAComputer = cms.ESProducer("GenericMVAJetTagESProducer",
    useCategories = cms.bool(False),
    calibrationRecord = cms.string('ImpactParameterMVA')
)


