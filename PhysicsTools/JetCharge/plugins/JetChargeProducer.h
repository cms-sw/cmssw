#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

#include "PhysicsTools/JetCharge/interface/JetCharge.h"

class JetChargeProducer : public edm::global::EDProducer<> {
    public:
        typedef reco::JetFloatAssociation::Container JetChargeCollection;

        explicit JetChargeProducer(const edm::ParameterSet &cfg) ;
        virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const;
    private:
        const edm::EDGetTokenT<reco::JetTracksAssociationCollection> srcToken_;
        const JetCharge     algo_;
};



