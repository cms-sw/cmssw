//
// $Id: TriggerObjectFilterByCollection.cc,v 1.1 2012/08/02 14:34:28 gpetrucc Exp $
//

/**
  \class     TriggerObjectFilterByCollection.h "MuonAnalysis/TagAndProbe/plugins/TriggerObjectFilterByCollection.h"
  \brief    Creates a filtered list of TriggerObjectStandAlone objects selecting by collection label 
            Inputs are:
                - a list of TriggerObjectStandAlone (param. "src")
                - a list of collections (param. "collections")
            Outputs are:
                - a list of TriggerObjectStandAlone
            
  \author   Giovanni Petrucciani
  \version  $Id: TriggerObjectFilterByCollection.cc,v 1.1 2012/08/02 14:34:28 gpetrucc Exp $
*/

#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"


class TriggerObjectFilterByCollection : public edm::EDProducer {
    public:
        explicit TriggerObjectFilterByCollection(const edm::ParameterSet & iConfig);
        virtual ~TriggerObjectFilterByCollection() { }

        virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
        edm::InputTag src_;
        std::vector<std::string> collections_;
};

TriggerObjectFilterByCollection::TriggerObjectFilterByCollection(const edm::ParameterSet & iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")), 
    collections_(iConfig.getParameter<std::vector<std::string> >("collections")) 
{
    produces<std::vector<pat::TriggerObjectStandAlone> >();
    for (unsigned int i = 0, n = collections_.size(); i < n; ++i) {
        std::string &c = collections_[i];
        int numsc = std::count(c.begin(), c.end(), ':');
        if      (numsc == 1) c.push_back(':');
        else if (numsc == 2) c.append("::");
    }
}

void 
TriggerObjectFilterByCollection::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;

    Handle<std::vector<pat::TriggerObjectStandAlone> > src;
    iEvent.getByLabel(src_, src);

    std::auto_ptr<std::vector<pat::TriggerObjectStandAlone> > out(new std::vector<pat::TriggerObjectStandAlone>());
    out->reserve(src->size());
    for (std::vector<pat::TriggerObjectStandAlone>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        const std::string &coll = it->collection();
        bool found = false;
        for (std::vector<std::string>::const_iterator ic = collections_.begin(), ec = collections_.end(); ic != ec; ++ic) {
            if (strncmp(coll.c_str(), ic->c_str(), ic->size()) == 0) { found = true; break; }
        }
        if (found) out->push_back(*it);
    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerObjectFilterByCollection);
