#include "PhysicsTools/JetCharge/plugins/JetChargeProducer.h"

JetChargeProducer::JetChargeProducer(const edm::ParameterSet &cfg) :
srcToken_(consumes<reco::JetTracksAssociationCollection>(cfg.getParameter<edm::InputTag>("src"))),
algo_(cfg) {
    produces<JetChargeCollection>();
}

void JetChargeProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
    edm::Handle<reco::JetTracksAssociationCollection> hJTAs;
    iEvent.getByToken(srcToken_, hJTAs);
    typedef reco::JetTracksAssociationCollection::const_iterator IT;
    typedef edm::RefToBase<reco::Jet>  JetRef;

    if (hJTAs->keyProduct().isNull()) {
        // need to work around this bug someway, altough it's not stricly my fault
        std::auto_ptr<JetChargeCollection> ret(new JetChargeCollection());
        iEvent.put(ret);
        return;
    }
    std::auto_ptr<JetChargeCollection> ret(new JetChargeCollection(hJTAs->keyProduct()));
    for (IT it = hJTAs->begin(), ed = hJTAs->end(); it != ed; ++it) {
        const JetRef &jet = it->first;
        const reco::TrackRefVector &tracks = it->second;
        float  val = static_cast<float>( algo_.charge(jet->p4(), tracks) );
        reco::JetFloatAssociation::setValue(*ret, jet, val);
    }

    iEvent.put(ret);
}
