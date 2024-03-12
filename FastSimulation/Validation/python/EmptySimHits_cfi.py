import FWCore.ParameterSet.Config as cms

emptySimHits = cms.EDProducer(
    'EmptySimHits',
    pSimHitInstanceLabels = cms.vstring(""),
    pCaloHitInstanceLabels = cms.vstring("")
    )

# foo bar baz
# RP7A80WfY4wz9
