#include "PhysicsTools/PatAlgos/plugins/AnythingToValueMap.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "PhysicsTools/PatUtils/interface/SimpleJetTrackAssociator.h"
#include "PhysicsTools/JetCharge/interface/JetCharge.h"

#include <algorithm>

class JetTracksAssociationAdapter {
public:
    typedef edm::View<reco::Jet>   Collection;
    typedef reco::TrackRefVector   value_type;

    // constructor
    JetTracksAssociationAdapter(const edm::ParameterSet &cfg) : 
        trackLabel_(cfg.getParameter<edm::InputTag>("tracks")),
        associator_ ( cfg.getParameter<double> ("deltaR"),
                      cfg.getParameter<int32_t>("minHits"),
                      cfg.getParameter<double> ("maxNormChi2") ) {}

    bool init(const edm::Event &iEvent) { 
        iEvent.getByLabel(trackLabel_, tracks_); 
        return !tracks_.failedToGet();
    }
    std::string label() { return ""; } 

    void run(const Collection &collection, std::vector<value_type> &ret) {
        ret.resize(collection.size());
        for (size_t i = 0, n = collection.size(); i < n; ++i) {
            associator_.associate(collection[i].momentum(), *tracks_, ret[i]);
        }
    }

private:
    edm::InputTag trackLabel_;
    edm::Handle<edm::View<reco::Track> > tracks_; 
    helper::SimpleJetTrackAssociator associator_;
};


class JetChargeAdapter {
public:
    typedef edm::View<reco::Jet>   Collection;
    typedef float                  value_type;

    // constructor
    JetChargeAdapter(const edm::ParameterSet &cfg) : 
        trackLabel_(cfg.getParameter<edm::InputTag>("jetTracksAssociation")),
        computer_ ( cfg ) {}

    bool init(const edm::Event &iEvent) { 
        iEvent.getByLabel(trackLabel_, tracks_); 
        return !tracks_.failedToGet();
    }
    std::string label() { return ""; } 

    void run(const Collection &collection, std::vector<value_type> &ret) {
        ret.reserve(collection.size());
        for (size_t i = 0, n = collection.size(); i < n; ++i) {
            float charge = computer_.charge( collection[i].p4(), (*tracks_)[collection.refAt(i)] );
            ret.push_back( charge ); 
        }
    }

private:
    edm::InputTag trackLabel_;
    edm::Handle<edm::ValueMap<reco::TrackRefVector> > tracks_; 
    JetCharge computer_;
};


typedef pat::helper::AnythingToValueMap<JetTracksAssociationAdapter> JetTracksAssociationValueMap;
typedef pat::helper::AnythingToValueMap<JetChargeAdapter> JetChargeValueMap;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetTracksAssociationValueMap);
DEFINE_FWK_MODULE(JetChargeValueMap);
