#include "PhysicsTools/JetCharge/plugins/JetChargeProducer.h"

JetChargeProducer::JetChargeProducer(const edm::ParameterSet &cfg) :
src_(cfg.getParameter<edm::InputTag>("src")),
algo_(cfg) {
    produces<reco::JetChargeCollection>();
}

void JetChargeProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
    edm::Handle<reco::JetTracksAssociationCollection> hJTAs;
    iEvent.getByLabel(src_, hJTAs);
    typedef reco::JetTracksAssociationCollection::const_iterator IT;
    typedef edm::RefToBase<reco::Jet>  JetRef;

    std::auto_ptr<reco::JetChargeCollection> ret(new reco::JetChargeCollection());
    ret->reserve(hJTAs->size());
    for (IT it = hJTAs->begin(), ed = hJTAs->end(); it != ed; ++it) {
        const JetRef &jet = it->first;
        const reco::TrackRefVector &tracks = it->second;
        float  val = static_cast<float>( algo_.charge(jet->p4(), tracks) );
        ret->push_back(reco::JetChargePair(jet, val));
    }

    iEvent.put(ret);        
}
