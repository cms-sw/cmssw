#include "PhysicsTools/PatAlgos/plugins/AnythingToValueMap.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

namespace pat { namespace helper {
class JetDiscriminatorAdaptor {
    public:
        typedef float                value_type;
        typedef edm::View<reco::Jet> Collection;

        JetDiscriminatorAdaptor(const edm::InputTag &in, const edm::ParameterSet & iConfig) :
            in_(in), label_(in.label() + in.instance()) { }

        const std::string & label() { return label_; }
        
        bool run(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) ;
    private:
        edm::InputTag in_;
        std::string label_;
};
    
bool JetDiscriminatorAdaptor::run(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) {
    edm::Handle<reco::JetFloatAssociation::Container> handle;
    iEvent.getByLabel(in_, handle);
    if (handle.failedToGet()) return false;

    for (size_t i = 0, n = coll.size(); i < n; ++i) {
        ret.push_back( (*handle)[ coll.refAt(i) ] );
    }
    return true;    
}

// ====================================================================================================================
class JetBaseTagInfoAdaptor {
    public:
        typedef edm::Ptr<reco::BaseTagInfo>  value_type;
        typedef edm::View<reco::Jet>         Collection;

        JetBaseTagInfoAdaptor(const edm::InputTag &in, const edm::ParameterSet & iConfig) :
            in_(in), label_(in.label() + in.instance()) { }

        const std::string & label() { return label_; }

        bool run(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) ;
    private:
        edm::InputTag in_;
        std::string label_;

};
bool JetBaseTagInfoAdaptor::run(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) {
    edm::Handle<edm::View<reco::BaseTagInfo> > handle;
    iEvent.getByLabel(in_, handle);
    if (handle.failedToGet()) return false;

    ret.resize(coll.size());
    for (size_t i = 0, n = coll.size(), n2 = handle->size(); i < n; ++i) {
        edm::RefToBase<reco::Jet> jetRef = coll.refAt(i);
        if ( (i < n2) &&( (*handle)[i].jet() == jetRef )) { // first try direct search
            ret[i] = handle->ptrAt(i);
        } else {
            for (edm::View<reco::BaseTagInfo>::const_iterator it = handle->begin(), ed = handle->end(); it != ed; ++it) {
                if (it->jet() == jetRef) {
                    ret[i] = handle->ptrAt(it - handle->begin());
                    break;
                }
            }
        }
    }
    return true;    
}
// ====================================================================================================================
template<typename TagInfo>
class JetTagInfoAdaptor {
    public:
        typedef edm::Ref<std::vector<TagInfo> > value_type; 
        typedef edm::View<reco::Jet>            Collection;

        JetTagInfoAdaptor(const edm::ParameterSet & iConfig) : in_(iConfig.getParameter<edm::InputTag>("tagInfos")) { }
        const std::string & label() { static const std::string empty; return empty; }
        bool init(const edm::Event &iEvent) { iEvent.getByLabel(in_, handle_); return ! handle_.failedToGet(); }
        void run(const Collection &coll, std::vector<value_type> &ret) ;
    private:
        edm::Handle<std::vector<TagInfo> > handle_;
        edm::InputTag in_;
};
template<typename TagInfo>
void JetTagInfoAdaptor<TagInfo>::run(const Collection &coll, std::vector<value_type> &ret) {
    ret.resize(coll.size());
    for (size_t i = 0, n = coll.size(), n2 = handle_->size(); i < n; ++i) {
        edm::RefToBase<reco::Jet> jetRef = coll.refAt(i);
        if ( (i < n2) &&( (*handle_)[i].jet() == jetRef )) { // first try direct search
            ret[i] = value_type(handle_, i);
        } else {
            for (typename std::vector<TagInfo>::const_iterator it = handle_->begin(), ed = handle_->end(); it != ed; ++it) {
                if (it->jet() == jetRef) {
                    ret[i] = value_type(handle_, it - handle_->begin());
                    break;
                }
            }
        }
    }
}



// ====================================================================================================================
typedef ManyThingsToValueMaps<JetDiscriminatorAdaptor> MultipleDiscriminatorsToValueMaps;
typedef ManyThingsToValueMaps<JetBaseTagInfoAdaptor>   MultipleTagInfosToValueMaps;
//typedef AnythingToValueMap<JetTagInfoAdaptor<reco::TrackIPTagInfo> >         TrackIPTagInfoToValueMap;
//typedef AnythingToValueMap<JetTagInfoAdaptor<reco::SoftLeptonTagInfo> >      SoftLeptonTagInfoToValueMap;
//typedef AnythingToValueMap<JetTagInfoAdaptor<reco::SecondaryVertexTagInfo> > SecondaryVertexTagInfoToValueMap;

}} // namespaces


#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat::helper;
DEFINE_FWK_MODULE(MultipleDiscriminatorsToValueMaps);
DEFINE_FWK_MODULE(MultipleTagInfosToValueMaps);
//DEFINE_FWK_MODULE(TrackIPTagInfoToValueMap);
//DEFINE_FWK_MODULE(SoftLeptonTagInfoToValueMap);
//DEFINE_FWK_MODULE(SecondaryVertexTagInfoToValueMap);
