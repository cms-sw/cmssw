import FWCore.ParameterSet.Config as cms

## Compute JET Charge
patJetCharge = cms.EDProducer("JetChargeProducer",
    src = cms.InputTag("ak4JetTracksAssociatorAtVertexPF"), ## a reco::JetTracksAssociation::Container
    # -- JetCharge parameters --
    var = cms.string('Pt'),
    exp = cms.double(1.0)
)

# removed for testing and final cleanup
# patJetTracksCharge = cms.Sequence(patAK5CaloJetCharge)
