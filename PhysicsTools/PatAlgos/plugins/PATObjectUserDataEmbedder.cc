#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataMerger.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

namespace {
  template <typename T>
  bool equal(const T &lhs, const T &rhs) {
    throw cms::Exception("NotImplimented")
        << " equal in PATObjectUserDataEmbedder is not implimented for objects of type " << typeid(lhs).name()
        << " and thus their src coll must be the same collection the valuemaps are keyed off";
  }
  template <>
  bool equal(const reco::GsfElectron &lhs, const reco::GsfElectron &rhs) {
    return lhs.superCluster()->seed()->seed().rawId() == rhs.superCluster()->seed()->seed().rawId();
  }
  template <>
  bool equal(const reco::Photon &lhs, const reco::Photon &rhs) {
    return lhs.superCluster()->seed()->seed().rawId() == rhs.superCluster()->seed()->seed().rawId();
  }

}  // namespace

namespace pat {

  namespace helper {

    struct AddUserIntFromBool {
      typedef bool value_type;
      typedef edm::ValueMap<value_type> product_type;
      template <typename ObjectType>
      void addData(ObjectType &obj, const std::string &key, const value_type &val) {
        obj.addUserInt(key, val);
      }
    };

    template <typename T, typename TParent, typename TProd>
    edm::Ptr<TParent> getPtrForProd(edm::Ptr<T> ptr,
                                    const std::vector<edm::Handle<edm::View<TParent>>> &parentColls,
                                    const TProd &prod) {
      if (prod.contains(ptr.id())) {
        return edm::Ptr<TParent>(ptr);
      } else {
        for (const auto &parentColl : parentColls) {
          if (parentColl.isValid() && prod.contains(parentColl.id())) {
            for (size_t indx = 0; indx < parentColl->size(); indx++) {
              if (equal<TParent>(*ptr, (*parentColl)[indx])) {
                return edm::Ptr<TParent>(parentColl, indx);
              }
            }
            //note this assumes that another parent coll isnt in the value map
            //it if its, it'll return null when the other one might work
            return edm::Ptr<TParent>(parentColl.id());
          }
        }
      }
      throw cms::Exception("ConfigurationError")
          << "When accessing value maps in PATObjectUserDataEmbedder, the collection the valuemap was keyed off is not "
             "either the input src or listed in one of the parentSrcs";
    }

    template <typename A>
    class NamedUserDataLoader {
    public:
      NamedUserDataLoader(const edm::ParameterSet &iConfig, const std::string &main, edm::ConsumesCollector &&cc) {
        if (iConfig.existsAs<edm::ParameterSet>(main)) {
          edm::ParameterSet const &srcPSet = iConfig.getParameter<edm::ParameterSet>(main);
          for (const std::string &label : srcPSet.getParameterNamesForType<edm::InputTag>()) {
            labelsAndTokens_.emplace_back(
                label, cc.consumes<typename A::product_type>(srcPSet.getParameter<edm::InputTag>(label)));
          }
        }
      }
      template <typename T, typename TParent = T>
      void addData(const edm::Event &iEvent,
                   const std::vector<edm::Ptr<T>> &ptrs,
                   std::vector<edm::Handle<edm::View<TParent>>> parents,
                   std::vector<T> &out) const {
        A adder;
        unsigned int n = ptrs.size();
        edm::Handle<typename A::product_type> handle;
        for (const auto &pair : labelsAndTokens_) {
          iEvent.getByToken(pair.second, handle);
          for (unsigned int i = 0; i < n; ++i) {
            auto ptr = getPtrForProd(ptrs[i], parents, *handle);
            adder.addData(out[i], pair.first, (*handle)[ptr]);
          }
        }
      }

    private:
      std::vector<std::pair<std::string, edm::EDGetTokenT<typename A::product_type>>> labelsAndTokens_;
    };  // class NamedUserDataLoader
  }     // namespace helper

  template <typename T, typename ParentType = T>
  class PATObjectUserDataEmbedder : public edm::stream::EDProducer<> {
  public:
    explicit PATObjectUserDataEmbedder(const edm::ParameterSet &iConfig)
        : src_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("src"))),
          userFloats_(iConfig, "userFloats", consumesCollector()),
          userInts_(iConfig, "userInts", consumesCollector()),
          userIntFromBools_(iConfig, "userIntFromBools", consumesCollector()),
          userCands_(iConfig, "userCands", consumesCollector()) {
      for (const auto &parentSrc : iConfig.getParameter<std::vector<edm::InputTag>>("parentSrcs")) {
        parentSrcs_.push_back(consumes<edm::View<ParentType>>(parentSrc));
      }
      produces<std::vector<T>>();
    }

    ~PATObjectUserDataEmbedder() override {}

    void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src");
      desc.add<std::vector<edm::InputTag>>("parentSrcs", std::vector<edm::InputTag>());
      for (auto &&what : {"userFloats", "userInts", "userIntFromBools", "userCands"}) {
        edm::ParameterSetDescription descNested;
        descNested.addWildcard<edm::InputTag>("*");
        desc.add<edm::ParameterSetDescription>(what, descNested);
      }
      if (typeid(T) == typeid(pat::Muon))
        descriptions.add("muonsWithUserData", desc);
      else if (typeid(T) == typeid(pat::Electron))
        descriptions.add("electronsWithUserData", desc);
      else if (typeid(T) == typeid(pat::Photon))
        descriptions.add("photonsWithUserData", desc);
      else if (typeid(T) == typeid(pat::Tau))
        descriptions.add("tausWithUserData", desc);
      else if (typeid(T) == typeid(pat::Jet))
        descriptions.add("jetsWithUserData", desc);
    }

  private:
    // configurables
    edm::EDGetTokenT<edm::View<T>> src_;
    //so valuemaps are keyed to a given collection so if we remake the objects,
    //are valuemaps are pointing to the wrong collection
    //this allows us to pass in past collections to try them to see if they are the ones
    //a valuemap is keyed to
    //note ParentType must inherit from T
    std::vector<edm::EDGetTokenT<edm::View<ParentType>>> parentSrcs_;

    helper::NamedUserDataLoader<pat::helper::AddUserFloat> userFloats_;
    helper::NamedUserDataLoader<pat::helper::AddUserInt> userInts_;
    helper::NamedUserDataLoader<pat::helper::AddUserIntFromBool> userIntFromBools_;
    helper::NamedUserDataLoader<pat::helper::AddUserCand> userCands_;
  };

}  // namespace pat

template <typename T, typename ParentType>
void pat::PATObjectUserDataEmbedder<T, ParentType>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<edm::View<T>> src;
  iEvent.getByToken(src_, src);

  std::vector<edm::Handle<edm::View<ParentType>>> parentSrcs;
  parentSrcs.reserve(parentSrcs_.size());
  for (const auto &src : parentSrcs_) {
    parentSrcs.push_back(iEvent.getHandle(src));
  }

  std::unique_ptr<std::vector<T>> out(new std::vector<T>());
  out->reserve(src->size());

  std::vector<edm::Ptr<T>> ptrs;
  ptrs.reserve(src->size());
  for (unsigned int i = 0, n = src->size(); i < n; ++i) {
    // copy by value, save the ptr
    out->push_back((*src)[i]);
    ptrs.push_back(src->ptrAt(i));
  }

  userFloats_.addData(iEvent, ptrs, parentSrcs, *out);
  userInts_.addData(iEvent, ptrs, parentSrcs, *out);
  userIntFromBools_.addData(iEvent, ptrs, parentSrcs, *out);
  userCands_.addData(iEvent, ptrs, parentSrcs, *out);

  iEvent.put(std::move(out));
}

typedef pat::PATObjectUserDataEmbedder<pat::Electron, reco::GsfElectron> PATElectronUserDataEmbedder;
typedef pat::PATObjectUserDataEmbedder<pat::Muon> PATMuonUserDataEmbedder;
typedef pat::PATObjectUserDataEmbedder<pat::Photon, reco::Photon> PATPhotonUserDataEmbedder;
typedef pat::PATObjectUserDataEmbedder<pat::Tau> PATTauUserDataEmbedder;
typedef pat::PATObjectUserDataEmbedder<pat::Jet> PATJetUserDataEmbedder;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATElectronUserDataEmbedder);
DEFINE_FWK_MODULE(PATMuonUserDataEmbedder);
DEFINE_FWK_MODULE(PATPhotonUserDataEmbedder);
DEFINE_FWK_MODULE(PATTauUserDataEmbedder);
DEFINE_FWK_MODULE(PATJetUserDataEmbedder);
