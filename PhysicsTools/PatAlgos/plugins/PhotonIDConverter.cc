#include "PhysicsTools/PatAlgos/plugins/AnythingToValueMap.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include "DataFormats/EgammaCandidates/interface/PhotonIDAssociation.h"

namespace pat { namespace helper {

class PhotonIDAdaptor {
    public:
        typedef reco::PhotonID           value_type; 
        typedef edm::View<reco::Photon>  Collection;

        PhotonIDAdaptor(const edm::ParameterSet & iConfig) : in_(iConfig.getParameter<edm::InputTag>("photonID")) { }
        const std::string & label() { static const std::string empty; return empty; }
        bool init(const edm::Event &iEvent) { iEvent.getByLabel(in_, handle_); return ! handle_.failedToGet(); }
        void run(const Collection &coll, std::vector<value_type> &ret) ;
    private:
        edm::Handle<reco::PhotonIDAssociationCollection> handle_;
        edm::InputTag in_;
};

void PhotonIDAdaptor::run(const Collection &coll, std::vector<value_type> &ret) {
    ret.resize(coll.size());
    for (size_t i = 0, n = coll.size(); i < n; ++i) {
        reco::PhotonRef photonRef = coll.refAt(i).castTo<reco::PhotonRef>();
        reco::PhotonIDAssociationCollection::const_iterator match = handle_->find(photonRef);
        if (match != handle_->end()) {
            ret[i] = * match->val;
        } 
    }
}

// ====================================================================================================================
typedef AnythingToValueMap<PhotonIDAdaptor>  PhotonIDConverter;

}} // namespaces


#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat::helper;
DEFINE_FWK_MODULE(PhotonIDConverter);
