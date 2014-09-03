#ifndef RecoAlgos_ObjectSelector_h
#define RecoAlgos_ObjectSelector_h
/** \class ObjectSelector
 *
 * selects a subset of a collection.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 * $Id: ObjectSelector.h,v 1.3 2010/02/20 20:55:27 wmtan Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/UtilAlgos/interface/NonNullNumberSelector.h"
#include "CommonTools/UtilAlgos/interface/StoreManagerTrait.h"
#include "CommonTools/UtilAlgos/interface/SelectedOutputCollectionTrait.h"
#include "CommonTools/UtilAlgos/interface/NullPostProcessor.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>

template<typename Selector,
         typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<typename Selector::collection>::type,
	 typename SizeSelector = NonNullNumberSelector,
	 typename PostProcessor = ::helper::NullPostProcessor<OutputCollection, edm::EDFilter>,
	 typename StoreManager = typename ::helper::StoreManagerTrait<OutputCollection, edm::EDFilter>::type,
	 typename Base = typename ::helper::StoreManagerTrait<OutputCollection, edm::EDFilter>::base,
	 typename Init = typename ::reco::modules::EventSetupInit<Selector>::type
	 >
class ObjectSelector : public Base {
public:
  /// constructor
  // ObjectSelector()=default;
  explicit ObjectSelector(const edm::ParameterSet & cfg) :
    Base(cfg),
    srcToken_( this-> template consumes<typename Selector::collection>(cfg.template getParameter<edm::InputTag>("src"))),
    filter_(false),
    selector_(cfg, this->consumesCollector()),
    sizeSelector_(reco::modules::make<SizeSelector>(cfg)),
    postProcessor_(cfg, this->consumesCollector()) {
    const std::string filter("filter");
    std::vector<std::string> bools = cfg.template getParameterNamesForType<bool>();
    bool found = std::find(bools.begin(), bools.end(), filter) != bools.end();
    if (found) filter_ = cfg.template getParameter<bool>(filter);
    postProcessor_.init(* this);
   }
  /// destructor
  virtual ~ObjectSelector() { }

private:
  /// process one event
  bool filter(edm::Event& evt, const edm::EventSetup& es) {
    Init::init(selector_, evt, es);
    using namespace std;
    edm::Handle<typename Selector::collection> source;
    evt.getByToken(srcToken_, source);
    StoreManager manager(source);
    selector_.select(source, evt, es);
    manager.cloneAndStore(selector_.begin(), selector_.end(), evt);
    bool result = (! filter_ || sizeSelector_(manager.size()));
    edm::OrphanHandle<OutputCollection> filtered = manager.put(evt);
    postProcessor_.process(filtered, evt);
    return result;
  }
  /// source collection label
  edm::EDGetTokenT<typename Selector::collection> srcToken_;
  /// filter event
  bool filter_;
  /// Object collection selector
  Selector selector_;
  /// selected object collection size selector
  SizeSelector sizeSelector_;
  /// post processor
  PostProcessor postProcessor_;
};

#endif

