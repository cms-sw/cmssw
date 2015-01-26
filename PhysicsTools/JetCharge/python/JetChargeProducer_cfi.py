import FWCore.ParameterSet.Config as cms

jetChargeProducer = cms.EDProducer("JetChargeProducer",
    var = cms.string('Pt'),
    src = cms.InputTag("ak4JTA"),
    exp = cms.double(1.0)
)


