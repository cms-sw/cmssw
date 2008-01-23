#include "PhysicsTools/PatAlgos/interface/AnythingToValueMap.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include <algorithm>

struct JetTagAdaptor {
    typedef edm::View<reco::Jet> Collection;
    typedef float value_type;

    edm::InputTag tagTag_;
    edm::Handle<std::vector<reco::JetTag> > tagHandle_;

    JetTagAdaptor(const edm::ParameterSet &cfg) : tagTag_(cfg.getParameter<edm::InputTag>("tags")) { }
    
    void init(const edm::Event &iEvent) { iEvent.getByLabel(tagTag_, tagHandle_); }
    std::string label() { return tagTag_.label() + tagTag_.instance(); } 

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

typedef pat::helper::AnythingToValueMap<JetTagAdaptor> JetTagToValueMapFloat;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetTagToValueMapFloat);
