#ifndef RecoAlgos_ObjectSelectorBase_h
#define RecoAlgos_ObjectSelectorBase_h
/** \class ObjectSelectorBase
 *
 * selects a subset of a collection.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 * $Id: ObjectSelectorBase.h,v 1.3 2010/02/20 20:55:27 wmtan Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>

template <typename Selector,
          typename OutputCollection,
          typename SizeSelector,
          typename PostProcessor,
          typename StoreManager,
          typename Base,
          typename Init>
class ObjectSelectorBase : public Base {
public:
  /// constructor
  // ObjectSelectorBase()=default;
  explicit ObjectSelectorBase(const edm::ParameterSet& cfg)
      : Base(cfg),
        srcToken_(
            this->template consumes<typename Selector::collection>(cfg.template getParameter<edm::InputTag>("src"))),
        filter_(false),
        throwOnMissing_(cfg.template getUntrackedParameter<bool>("throwOnMissing", true)),
        selectorInit_(this->consumesCollector()),
        selector_(cfg, this->consumesCollector()),
        sizeSelector_(reco::modules::make<SizeSelector>(cfg)),
        postProcessor_(cfg, this->consumesCollector()) {
    const std::string filter("filter");
    std::vector<std::string> bools = cfg.template getParameterNamesForType<bool>();
    bool found = std::find(bools.begin(), bools.end(), filter) != bools.end();
    if (found)
      filter_ = cfg.template getParameter<bool>(filter);
    postProcessor_.init(*this);
  }
  /// destructor
  ~ObjectSelectorBase() override {}

private:
  /// process one event
  bool filter(edm::Event& evt, const edm::EventSetup& es) override {
    selectorInit_.init(selector_, evt, es);
    edm::Handle<typename Selector::collection> source;
    if (!throwOnMissing_ && !source.isValid()) {
      return !filter_;
    }
    evt.getByToken(srcToken_, source);
    StoreManager manager(source);
    selector_.select(source, evt, es);
    manager.cloneAndStore(selector_.begin(), selector_.end(), evt);
    bool result = (!filter_ || sizeSelector_(manager.size()));
    edm::OrphanHandle<OutputCollection> filtered = manager.put(evt);
    postProcessor_.process(filtered, evt);
    return result;
  }
  /// source collection label
  edm::EDGetTokenT<typename Selector::collection> srcToken_;
  /// filter event
  bool filter_;
  /// trhow on missing
  bool throwOnMissing_;
  /// Object collection selector
  Init selectorInit_;
  Selector selector_;
  /// selected object collection size selector
  SizeSelector sizeSelector_;
  /// post processor
  PostProcessor postProcessor_;
};

#endif
