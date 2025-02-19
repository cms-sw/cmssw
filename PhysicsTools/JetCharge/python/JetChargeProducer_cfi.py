import FWCore.ParameterSet.Config as cms

jetChargeProducer = cms.EDProducer("JetChargeProducer",
    var = cms.string('Pt'),
    src = cms.InputTag("ic5JetTracksAssociatorAtVertex"),
    exp = cms.double(1.0)
)


