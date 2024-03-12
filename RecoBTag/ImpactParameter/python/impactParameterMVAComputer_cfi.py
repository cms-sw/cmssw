import FWCore.ParameterSet.Config as cms

# impactParameterMVAComputer btag computer
impactParameterMVAComputer = cms.ESProducer("GenericMVAJetTagESProducer",
    useCategories = cms.bool(False),
    calibrationRecord = cms.string('ImpactParameterMVA'),
    recordLabel = cms.string('')
)
# foo bar baz
# oCCp5Axmyetbb
# HxxurEY3bYOjF
