#ifndef FWCore_Framework_ThinningProducer_h
#define FWCore_Framework_ThinningProducer_h

/** \class edm::ThinningProducer
\author W. David Dagenhart, created 11 June 2014
*/

#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Common/interface/fillCollectionForThinning.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>
#include <optional>
#include <type_traits>

namespace edm {

  class EventSetup;

  namespace detail {
    template <typename T>
    struct IsStdOptional {
      static constexpr bool value = false;
    };
    template <typename T>
    struct IsStdOptional<std::optional<T>> {
      static constexpr bool value = true;
    };

    template <typename Item, typename Selector, typename Collection>
    void fillCollectionForThinning(Item const& item,
                                   Selector& selector,
                                   unsigned int iIndex,
                                   Collection& output,
                                   ThinnedAssociation& association) {
      using SelectorChooseReturnType = decltype(selector.choose(0U, std::declval<Item const&>()));
      constexpr bool isSlimming = detail::IsStdOptional<SelectorChooseReturnType>::value;
      if constexpr (isSlimming) {
        std::optional<typename SelectorChooseReturnType::value_type> obj = selector.choose(iIndex, item);
        if (obj.has_value()) {
          // move to support std::unique_ptr<T> with edm::OwnVector<T> or std::vector<unique_ptr<T>>
          output.push_back(std::move(*obj));
          association.push_back(iIndex);
        }
      } else {
        if (selector.choose(iIndex, item)) {
          output.push_back(item);
          association.push_back(iIndex);
        }
      }
    }

  }  // namespace detail

  template <typename Collection, typename Selector>
  class ThinningProducer : public stream::EDProducer<> {
  public:
    explicit ThinningProducer(ParameterSet const& pset);
    ~ThinningProducer() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

    void produce(Event& event, EventSetup const& eventSetup) override;

    void registerThinnedAssociations(ProductRegistry const& productRegistry,
                                     ThinnedAssociationsHelper& thinnedAssociationsHelper) override;

  private:
    edm::propagate_const<std::unique_ptr<Selector>> selector_;
    edm::EDGetTokenT<Collection> inputToken_;
    edm::InputTag inputTag_;
    edm::EDPutTokenT<Collection> outputToken_;
    edm::EDPutTokenT<ThinnedAssociation> thinnedOutToken_;

    using SelectorChooseReturnType =
        decltype(selector_->choose(0U, std::declval<typename detail::ElementType<Collection>::type const>()));
    static constexpr bool isSlimming = detail::IsStdOptional<SelectorChooseReturnType>::value;
    static_assert(
        std::is_same_v<SelectorChooseReturnType, bool> || isSlimming,
        "Selector::choose() must return bool (for pure thinning) or std::optional<ElementType> (for slimming)");
  };

  template <typename Collection, typename Selector>
  ThinningProducer<Collection, Selector>::ThinningProducer(ParameterSet const& pset)
      : selector_(new Selector(pset, consumesCollector())) {
    inputTag_ = pset.getParameter<InputTag>("inputTag");
    inputToken_ = consumes<Collection>(inputTag_);

    outputToken_ = produces<Collection>();
    thinnedOutToken_ = produces<ThinnedAssociation>();
  }

  template <typename Collection, typename Selector>
  ThinningProducer<Collection, Selector>::~ThinningProducer() {}

  template <typename Collection, typename Selector>
  void ThinningProducer<Collection, Selector>::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Produces thinned collections and associations to them");
    desc.add<edm::InputTag>("inputTag");
    Selector::fillPSetDescription(desc);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename Collection, typename Selector>
  void ThinningProducer<Collection, Selector>::produce(Event& event, EventSetup const& eventSetup) {
    auto inputCollection = event.getHandle(inputToken_);

    edm::Event const& constEvent = event;
    selector_->preChoose(inputCollection, constEvent, eventSetup);

    Collection thinnedCollection;
    ThinnedAssociation thinnedAssociation;

    unsigned int iIndex = 0;
    for (auto iter = inputCollection->begin(), iterEnd = inputCollection->end(); iter != iterEnd; ++iter, ++iIndex) {
      using namespace detail;
      fillCollectionForThinning(*iter, *selector_, iIndex, thinnedCollection, thinnedAssociation);
    }
    selector_->reset();

    OrphanHandle<Collection> orphanHandle = event.emplace(outputToken_, std::move(thinnedCollection));

    thinnedAssociation.setParentCollectionID(inputCollection.id());
    thinnedAssociation.setThinnedCollectionID(orphanHandle.id());
    event.emplace(thinnedOutToken_, std::move(thinnedAssociation));
  }

  template <typename Collection, typename Selector>
  void ThinningProducer<Collection, Selector>::registerThinnedAssociations(
      ProductRegistry const& productRegistry, ThinnedAssociationsHelper& thinnedAssociationsHelper) {
    BranchID associationID;
    BranchID thinnedCollectionID;

    // If the InputTag does not specify the process name, it is
    // possible that there will be more than one match found below.
    // For a particular event only one match is correct and the
    // others will be false. It even possible for some events one
    // match is correct and for others another is correct. This is
    // a side effect of the lookup mechanisms when the process name
    // is not specified.
    // When using the registry this generates one would have to
    // check the ProductIDs in ThinnedAssociation product to get
    // the correct association. This ambiguity will probably be
    // rare and possibly never occur in practice.
    std::vector<BranchID> parentCollectionIDs;

    ProductRegistry::ProductList const& productList = productRegistry.productList();
    for (auto const& product : productList) {
      BranchDescription const& desc = product.second;
      if (desc.dropped()) {
        // Dropped branch does not have type information, but they can
        // be ignored here because all of the parent/thinned/association
        // branches are expected to be present
        continue;
      }
      if (desc.unwrappedType().typeInfo() == typeid(Collection)) {
        if (desc.produced() && desc.moduleLabel() == moduleDescription().moduleLabel() &&
            desc.productInstanceName().empty()) {
          thinnedCollectionID = desc.branchID();
        }
        if (desc.moduleLabel() == inputTag_.label() && desc.productInstanceName() == inputTag_.instance()) {
          if (inputTag_.willSkipCurrentProcess()) {
            if (!desc.produced()) {
              parentCollectionIDs.push_back(desc.branchID());
            }
          } else if (inputTag_.process().empty() || inputTag_.process() == desc.processName()) {
            if (desc.produced()) {
              parentCollectionIDs.push_back(desc.originalBranchID());
            } else {
              parentCollectionIDs.push_back(desc.branchID());
            }
          }
        }
      }
      if (desc.produced() && desc.unwrappedType().typeInfo() == typeid(ThinnedAssociation) &&
          desc.moduleLabel() == moduleDescription().moduleLabel() && desc.productInstanceName().empty()) {
        associationID = desc.branchID();
      }
    }
    if (parentCollectionIDs.empty()) {
      // This could happen if the input collection was dropped. Go ahead and add
      // an entry and let the exception be thrown only if the module is run (when
      // it cannot find the product).
      thinnedAssociationsHelper.addAssociation(BranchID(), associationID, thinnedCollectionID, isSlimming);
    } else {
      for (auto const& parentCollectionID : parentCollectionIDs) {
        thinnedAssociationsHelper.addAssociation(parentCollectionID, associationID, thinnedCollectionID, isSlimming);
      }
    }
  }
}  // namespace edm
#endif
