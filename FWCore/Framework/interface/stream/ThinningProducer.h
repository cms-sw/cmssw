#ifndef FWCore_Framework_ThinningProducer_h
#define FWCore_Framework_ThinningProducer_h

/** \class edm::ThinningProducer
\author W. David Dagenhart, created 11 June 2014
*/

#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>

namespace edm {

  class EventSetup;

  template <typename Collection, typename Selector>
  class ThinningProducer : public stream::EDProducer<> {
  public:
    explicit ThinningProducer(ParameterSet const& pset);
    virtual ~ThinningProducer();

    static void fillDescriptions(ConfigurationDescriptions & descriptions);

    virtual void produce(Event& event, EventSetup const& eventSetup) override;

    virtual void registerThinnedAssociations(ProductRegistry const& productRegistry,
                                             ThinnedAssociationsHelper& thinnedAssociationsHelper) override;
  private:
    std::unique_ptr<Selector> selector_;
    edm::EDGetTokenT<Collection> inputToken_;
    edm::InputTag inputTag_;
  };

  template <typename Collection, typename Selector>
  ThinningProducer<Collection, Selector>::
  ThinningProducer(ParameterSet const& pset) :
    selector_(new Selector(pset, consumesCollector())) {

    inputTag_ = pset.getParameter<InputTag>("inputTag");
    inputToken_ = consumes<Collection>(inputTag_);

    produces<Collection>();
    produces<ThinnedAssociation>();
  }

  template <typename Collection, typename Selector>
  ThinningProducer<Collection, Selector>::
  ~ThinningProducer() {}

  template <typename Collection, typename Selector>
  void ThinningProducer<Collection, Selector>::
  fillDescriptions(ConfigurationDescriptions & descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Produces thinned collections and associations to them");
    desc.add<edm::InputTag>("inputTag");
    Selector::fillDescription(desc);
    descriptions.addDefault(desc);
  }

  template <typename Collection, typename Selector>
  void ThinningProducer<Collection, Selector>::
  produce(Event& event, EventSetup const& eventSetup) {

    edm::Handle<Collection> inputCollection;
    event.getByToken(inputToken_, inputCollection);

    edm::Event const& constEvent = event;
    selector_->preChoose(inputCollection, constEvent, eventSetup);

    std::auto_ptr<Collection> thinnedCollection(new Collection);
    std::auto_ptr<ThinnedAssociation> thinnedAssociation(new ThinnedAssociation);

    unsigned int iIndex = 0;
    for(auto iter = inputCollection->begin(), iterEnd = inputCollection->end();
        iter != iterEnd; ++iter, ++iIndex) {
      if(selector_->choose(iIndex, *iter)) {
        thinnedCollection->push_back(*iter);
        thinnedAssociation->push_back(iIndex);
      }
    }
    OrphanHandle<Collection> orphanHandle = event.put(thinnedCollection);

    thinnedAssociation->setParentCollectionID(inputCollection.id());
    thinnedAssociation->setThinnedCollectionID(orphanHandle.id());
    event.put(thinnedAssociation);
  }

  template <typename Collection, typename Selector>
  void ThinningProducer<Collection, Selector>::
  registerThinnedAssociations(ProductRegistry const& productRegistry,
                              ThinnedAssociationsHelper& thinnedAssociationsHelper) {

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
    for(auto const& product : productList) {
      BranchDescription const& desc = product.second;
      if(desc.unwrappedType().typeInfo() == typeid(Collection) ) {
        if(desc.produced() &&
           desc.moduleLabel() == moduleDescription().moduleLabel() &&
           desc.productInstanceName().empty()) {

          thinnedCollectionID = desc.branchID();
        }
        if(desc.moduleLabel() == inputTag_.label() &&
           desc.productInstanceName() == inputTag_.instance()) {
          if(inputTag_.willSkipCurrentProcess()) {
            if(!desc.produced()) {
              parentCollectionIDs.push_back(desc.branchID());
            }
          } else if (inputTag_.process().empty() || inputTag_.process() == desc.processName()) {
            if(desc.produced()) {
              parentCollectionIDs.push_back(desc.originalBranchID());
            } else {
              parentCollectionIDs.push_back(desc.branchID());
            }
          }
        }
      }
      if(desc.produced() &&
         desc.unwrappedType().typeInfo() == typeid(ThinnedAssociation) &&
         desc.moduleLabel() == moduleDescription().moduleLabel() &&
         desc.productInstanceName().empty()) {

        associationID = desc.branchID();
      }
    }
    if(parentCollectionIDs.empty()) {
      // This could happen if the input collection was dropped. Go ahead and add
      // an entry and let the exception be thrown only if the module is run (when
      // it cannot find the product).
      thinnedAssociationsHelper.addAssociation(BranchID(), associationID, thinnedCollectionID);
    } else {
      for(auto const& parentCollectionID : parentCollectionIDs) {
        thinnedAssociationsHelper.addAssociation(parentCollectionID, associationID, thinnedCollectionID);
      }
    }
  }
}
#endif
