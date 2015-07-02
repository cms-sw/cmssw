#ifndef CommonTools_UtilAlgos_ObjectSelectorProducer_h
#define CommonTools_UtilAlgos_ObjectSelectorProducer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/StoreManagerTrait.h"
#include "CommonTools/UtilAlgos/interface/SelectedOutputCollectionTrait.h"
#include "CommonTools/UtilAlgos/interface/NullPostProcessor.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>

/**
 * This class template is like ObjectSelector, but it is an EDProducer
 * instead of EDFilter. Use case is that when the filter decisions are
 * ignored (cms.ignore in configuration or EDFilter returns always
 * true), EDProducers are better for the unscheduled mode than
 * EDFilters.
 */
template<typename Selector,
         typename OutputCollection,
         typename PostProcessor,
         typename StoreManager,
         typename Base,
         typename Init
         >
class ObjectSelectorProducer : public Base {
public:
  /// constructor
  explicit ObjectSelectorProducer(const edm::ParameterSet & cfg) :
    Base(cfg),
    srcToken_( this-> template consumes<typename Selector::collection>(cfg.template getParameter<edm::InputTag>("src"))),
    selector_(cfg, this->consumesCollector()),
    postProcessor_(cfg, this->consumesCollector()) {
    postProcessor_.init(* this);
   }
  /// destructor
  virtual ~ObjectSelectorProducer() { }

private:
  /// process one event
  void produce(edm::Event& evt, const edm::EventSetup& es) override {
    Init::init(selector_, evt, es);
    using namespace std;
    edm::Handle<typename Selector::collection> source;
    evt.getByToken(srcToken_, source);
    StoreManager manager(source);
    selector_.select(source, evt, es);
    manager.cloneAndStore(selector_.begin(), selector_.end(), evt);
    edm::OrphanHandle<OutputCollection> filtered = manager.put(evt);
    postProcessor_.process(filtered, evt);
  }
  /// source collection label
  edm::EDGetTokenT<typename Selector::collection> srcToken_;
  /// Object collection selector
  Selector selector_;
  /// post processor
  PostProcessor postProcessor_;
};


#endif
