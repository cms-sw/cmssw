#include "PhysicsTools/PatAlgos/plugins/AnythingToValueMap.h"
#include "DataFormats/BTauReco/interface/JetTagFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include <algorithm>

struct JetTagAdaptor {
    typedef edm::View<reco::Jet> Collection;
    typedef float value_type;

    edm::InputTag tagTag_;
    edm::Handle<std::vector<reco::JetTag> > tagHandle_;

    JetTagAdaptor(const edm::ParameterSet &cfg) : tagTag_(cfg.getParameter<edm::InputTag>("tags")) { }
    
    bool init(const edm::Event &iEvent) { iEvent.getByLabel(tagTag_, tagHandle_); return !tagHandle_.failedToGet(); }
    std::string label() { return ""; } 

    void run(const Collection &collection, std::vector<value_type> &ret) {
        size_t tagsize = tagHandle_->size();
        using namespace std;
        for (size_t i = 0, n = collection.size(); i < n; ++i) {
            double val = -10.0f;
            edm::RefToBase<reco::Jet> jet = collection.refAt(i);
            if ((i < tagsize) && ((*tagHandle_)[i].jet() == jet)) {
                val = (*tagHandle_)[i].discriminator();
            } else {
                for (std::vector<reco::JetTag>::const_iterator itt = tagHandle_->begin(), edt = tagHandle_->end();  itt != edt; ++itt) {
                    if (itt->jet() == jet) { 
                        val = itt->discriminator(); 
                        break; 
                    }
                }    
            }
            ret.push_back(val);
        }
    }
};

// Ugly, not needed in 20X
struct JetTagRefAdaptor {
    typedef edm::View<reco::Jet> Collection;
    typedef reco::JetTagRef value_type;

    edm::InputTag tagTag_;
    edm::Handle<std::vector<reco::JetTag> > tagHandle_;

    JetTagRefAdaptor(const edm::ParameterSet &cfg) : tagTag_(cfg.getParameter<edm::InputTag>("tags")) { }
    
    bool init(const edm::Event &iEvent) { iEvent.getByLabel(tagTag_, tagHandle_); return !tagHandle_.failedToGet(); }
    std::string label() { return ""; } 

    void run(const Collection &collection, std::vector<value_type> &ret) {
        size_t tagsize = tagHandle_->size();
        using namespace std;
        for (size_t i = 0, n = collection.size(); i < n; ++i) {
            value_type val;
            edm::RefToBase<reco::Jet> jet = collection.refAt(i);
            if ((i < tagsize) && ((*tagHandle_)[i].jet() == jet)) {
                val = value_type(tagHandle_, i); 
            } else {
                for (std::vector<reco::JetTag>::const_iterator itt = tagHandle_->begin(), edt = tagHandle_->end();  itt != edt; ++itt) {
                    if (itt->jet() == jet) { 
                        val = value_type(tagHandle_, itt - tagHandle_->begin()); 
                        break; 
                    }
                }    
            }
            ret.push_back(val);
        }
    }
};


typedef pat::helper::AnythingToValueMap<JetTagAdaptor> JetTagToValueMapFloat;
typedef pat::helper::AnythingToValueMap<JetTagRefAdaptor> JetTagToValueMapRef;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetTagToValueMapFloat);
DEFINE_FWK_MODULE(JetTagToValueMapRef);
