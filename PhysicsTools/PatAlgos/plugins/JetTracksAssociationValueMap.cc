#include "PhysicsTools/PatAlgos/plugins/AnythingToValueMap.h"
#include "PhysicsTools/JetCharge/interface/JetCharge.h"
#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/View.h"

#include <algorithm>

class JetTracksAssociationAdapter {
public:
    typedef edm::View<reco::Jet>   Collection;
    typedef reco::TrackRefVector   value_type;

    // constructor
    JetTracksAssociationAdapter(const edm::ParameterSet &cfg) : 
        jtaLabel_(cfg.getParameter<edm::InputTag>("tracks")),
        hasSelector_(cfg.exists("cut") && !cfg.getParameter<std::string>("cut").empty()),
        selector_(hasSelector_ ? cfg.getParameter<std::string>("cut") : "")  {}

    bool init(const edm::Event &iEvent) { 
        iEvent.getByLabel(jtaLabel_, tracks_); 
        return !tracks_.failedToGet();
    }
    std::string label() { return ""; } 

    void run(const Collection &collection, std::vector<value_type> &ret) {
        ret.resize(collection.size());
        for (size_t i = 0, n = collection.size(); i < n; ++i) {
            const value_type & tks =  (*tracks_)[collection.refAt(i)];
            if (hasSelector_) { // check each track
                for (value_type::const_iterator itk = tks.begin(), etk = tks.end(); itk != etk; ++itk) {
                    if (selector_(**itk)) ret[i].push_back(*itk);
                }
            } else { // bulk copy
                ret[i] = tks;
            }
        }
    }

private:
    edm::InputTag jtaLabel_;
    edm::Handle<reco::JetTracksAssociationCollection> tracks_; 
    bool hasSelector_;
    StringCutObjectSelector<reco::Track> selector_;
};


class JetChargeAdapter {
public:
    typedef edm::View<reco::Jet>   Collection;
    typedef float                  value_type;

    // constructor
    JetChargeAdapter(const edm::ParameterSet &cfg) : 
        jtaLabel_(cfg.getParameter<edm::InputTag>("jetTracksAssociation")),
        computer_ ( cfg ) {}

    bool init(const edm::Event &iEvent) { 
        iEvent.getByLabel(jtaLabel_, tracks_); 
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
    edm::InputTag jtaLabel_;
    edm::Handle<edm::ValueMap<reco::TrackRefVector> > tracks_; 
    JetCharge computer_;
};


typedef pat::helper::AnythingToValueMap<JetTracksAssociationAdapter> JetTracksAssociationValueMap;
typedef pat::helper::AnythingToValueMap<JetChargeAdapter> JetChargeValueMap;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetTracksAssociationValueMap);
DEFINE_FWK_MODULE(JetChargeValueMap);
