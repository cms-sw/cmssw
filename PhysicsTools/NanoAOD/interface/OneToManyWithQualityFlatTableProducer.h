#ifndef PhysicsTools_NanoAOD_OneToManyWithQualityFlatTableProducer_h
#define PhysicsTools_NanoAOD_OneToManyWithQualityFlatTableProducer_h

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

template <typename T>
struct OneToManyWithQualityTraits;

template <typename CKey,
          typename CVal,
          typename Q,
          typename Index,
          typename KeyRefProd,
          typename ValRefProd,
          typename KeyRef,
          typename ValRef>
struct OneToManyWithQualityTraits<
    edm::AssociationMap<edm::OneToManyWithQualityGeneric<CKey, CVal, Q, Index, KeyRefProd, ValRefProd, KeyRef, ValRef>>> {
  using key_collection_type = CKey;
  using val_collection_type = CVal;
  using quality_type = Q;
};

template <typename T>
class OneToManyWithQualityFlatTableProducer : public edm::stream::EDProducer<> {
public:
  using Traits = OneToManyWithQualityTraits<T>;
  using KeyCollection = typename Traits::key_collection_type;
  using ValCollection = typename Traits::val_collection_type;

  explicit OneToManyWithQualityFlatTableProducer(edm::ParameterSet const& params)
      : srcToken_(consumes<T>(params.getParameter<edm::InputTag>("src"))),
        keyToken_(consumes<KeyCollection>(params.getParameter<edm::InputTag>("keySrc"))),
        valToken_(consumes<ValCollection>(params.getParameter<edm::InputTag>("valSrc"))),
        name_(params.getParameter<std::string>("name")),
        doc_(params.getParameter<std::string>("doc")),
        linksName_(params.getParameter<std::string>("linksName")),
        linksDoc_(params.getParameter<std::string>("linksDoc")),
        scorePrecision_(params.getParameter<int>("scorePrecision")),
        skipNonExistingSrc_(params.getParameter<bool>("skipNonExistingSrc")) {
    produces<nanoaod::FlatTable>();
    produces<nanoaod::FlatTable>(linksName_ + "Table");
  }

  ~OneToManyWithQualityFlatTableProducer() override = default;

  void produce(edm::Event& iEvent, edm::EventSetup const&) override {
    auto const prod = iEvent.getHandle(srcToken_);
    auto const keyHandle = iEvent.getHandle(keyToken_);
    auto const valHandle = iEvent.getHandle(valToken_);

    size_t const N = keyHandle.isValid() ? keyHandle->size() : 0;
    std::vector<uint16_t> counts(N, 0);
    std::vector<uint16_t> offsets(N, 0);
    std::vector<uint32_t> flatIndex;
    std::vector<float> flatScore;

    if ((prod.isValid() && keyHandle.isValid() && valHandle.isValid()) || !skipNonExistingSrc_) {
      if (prod->refProd().key.id() != keyHandle.id()) {
        throw cms::Exception("Configuration")
            << "OneToManyWithQualityFlatTableProducer (" << name_ << "): keySrc ProductID " << keyHandle.id()
            << " does not match the map's internal key RefProd ProductID " << prod->refProd().key.id()
            << ". keySrc must point to the same collection the associator ran on.";
      }
      if (prod->refProd().val.id() != valHandle.id()) {
        throw cms::Exception("Configuration")
            << "OneToManyWithQualityFlatTableProducer (" << name_ << "): valSrc ProductID " << valHandle.id()
            << " does not match the map's internal val RefProd ProductID " << prod->refProd().val.id()
            << ". valSrc must point to the same collection the associator ran on.";
      }

      // The underlying map is std::map<index, vector<pair<index, Q>>>, hence iterates in
      // increasing key-index order. No KeyRef construction is needed, so this works uniformly
      // whether CKey is a concrete collection or an edm::View.
      for (auto it = prod->begin(), end = prod->end(); it != end; ++it) {
        unsigned int const keyIdx = it->key.key();
        if (keyIdx >= N) {
          throw cms::Exception("LogicError") << "OneToManyWithQualityFlatTableProducer (" << name_ << "): key index "
                                             << keyIdx << " is out of range for a key collection of size " << N << ".";
        }
        auto const& links = it->val;
        counts[keyIdx] = static_cast<uint16_t>(links.size());
        flatIndex.reserve(flatIndex.size() + links.size());
        flatScore.reserve(flatScore.size() + links.size());
        for (auto const& link : links) {
          flatIndex.push_back(static_cast<uint32_t>(link.first.key()));
          flatScore.push_back(static_cast<float>(link.second));
        }
      }
      uint16_t running = 0;
      for (size_t i = 0; i < N; ++i) {
        offsets[i] = running;
        running += counts[i];
      }
    }

    auto mainOut = std::make_unique<nanoaod::FlatTable>(N, name_, false);
    mainOut->template addColumn<uint16_t>("n" + linksName_, counts, "number of " + linksName_ + " entries");
    mainOut->template addColumn<uint16_t>("o" + linksName_, offsets, "offset into " + linksName_ + " sub-table");
    mainOut->setDoc(doc_);

    auto linksOut = std::make_unique<nanoaod::FlatTable>(flatIndex.size(), linksName_, false, false);
    linksOut->template addColumn<uint32_t>("index", flatIndex, "index in the val (target) collection");
    linksOut->template addColumn<float>("score", flatScore, "association quality score", scorePrecision_);
    linksOut->setDoc(linksDoc_);

    iEvent.put(std::move(mainOut));
    iEvent.put(std::move(linksOut), linksName_ + "Table");
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src")->setComment(
        "input edm::AssociationMap<edm::OneToManyWithQualityGeneric<...>> produced by an associator");
    desc.add<edm::InputTag>("keySrc")->setComment(
        "key (source) collection used by the associator; sizes the dense main table");
    desc.add<edm::InputTag>("valSrc")->setComment(
        "val (target) collection used by the associator; target of the stored indices");
    desc.add<std::string>("name")->setComment("name of the main per-key FlatTable");
    desc.add<std::string>("doc", "")->setComment("documentation for the main table");
    desc.add<std::string>("linksName")->setComment("name of the flattened sub-table (one row per association)");
    desc.add<std::string>("linksDoc", "")->setComment("documentation for the links sub-table");
    desc.add<int>("scorePrecision", 14)
        ->setComment("mantissa bits kept for the score column (<=0 = full float precision)");
    desc.add<bool>("skipNonExistingSrc", false)
        ->setComment("if true, emit empty tables when any input handle is invalid");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  edm::EDGetTokenT<T> srcToken_;
  edm::EDGetTokenT<KeyCollection> keyToken_;
  edm::EDGetTokenT<ValCollection> valToken_;
  std::string name_;
  std::string doc_;
  std::string linksName_;
  std::string linksDoc_;
  int scorePrecision_;
  bool skipNonExistingSrc_;
};

#endif
