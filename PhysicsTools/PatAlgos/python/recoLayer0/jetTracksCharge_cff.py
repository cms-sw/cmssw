import FWCore.ParameterSet.Config as cms

## Compute JET Charge
patJetCharge = cms.EDFilter("JetChargeProducer",
    src = cms.InputTag("ic5JetTracksAssociatorAtVertex"), ## a reco::JetTracksAssociation::Container
    # -- JetCharge parameters --
    var = cms.string('Pt'),
    exp = cms.double(1.0)
)

patJetTracksCharge = cms.Sequence(patJetCharge)
