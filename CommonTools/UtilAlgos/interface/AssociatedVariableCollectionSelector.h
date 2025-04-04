#ifndef CommonTools_UtilAlgos_AssociatedVariableCollectionSelector_h
#define CommonTools_UtilAlgos_AssociatedVariableCollectionSelector_h
/* \class AssociatedVariableCollectionSelector
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: AssociatedVariableCollectionSelector.h,v 1.2 2010/02/20 20:55:13 wmtan Exp $
 *
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CommonTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "CommonTools/UtilAlgos/interface/SelectedOutputCollectionTrait.h"
#include "CommonTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "DataFormats/Common/interface/getRef.h"

namespace reco {
  namespace modules {
    template <typename S>
    struct AssociatedVariableCollectionSelectorEventSetupInit;
  }
}  // namespace reco

template <typename InputCollection,
          typename VarCollection,
          typename Selector,
          typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<InputCollection>::type,
          typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
          typename RefAdder = typename ::helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
class AssociatedVariableCollectionSelector {
public:
  typedef InputCollection collection;
  typedef StoreContainer container;
  typedef Selector selector;
  typedef typename container::const_iterator const_iterator;
  AssociatedVariableCollectionSelector(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : varToken_(iC.consumes<VarCollection>(cfg.template getParameter<edm::InputTag>("var"))),
        select_(reco::modules::make<Selector>(cfg, iC)) {}
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select(const edm::Handle<InputCollection>& c, const edm::Event& evt, const edm::EventSetup&) {
    selected_.clear();
    edm::Handle<VarCollection> var;
    evt.getByToken(varToken_, var);
    for (size_t idx = 0; idx < c->size(); ++idx) {
      if (select_((*c)[idx], (*var)[edm::getRef(c, idx)]))
        addRef_(selected_, c, idx);
    }
  }

  static void fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<edm::InputTag>("var", edm::InputTag(""));
    Selector::fillPSetDescription(desc);
  }

private:
  edm::EDGetTokenT<VarCollection> varToken_;
  container selected_;
  selector select_;
  RefAdder addRef_;
  friend struct reco::modules::AssociatedVariableCollectionSelectorEventSetupInit<AssociatedVariableCollectionSelector>;
};

#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"

namespace reco {
  namespace modules {
    template <typename S>
    struct AssociatedVariableCollectionSelectorEventSetupInit {
      explicit AssociatedVariableCollectionSelectorEventSetupInit(edm::ConsumesCollector iC) : esi_(iC) {}

      void init(S& s, const edm::Event& evt, const edm::EventSetup& es) { esi_.init(s.select_, evt, es); }
      typedef typename EventSetupInit<typename S::selector>::type ESI;
      ESI esi_;
    };

    template <typename I, typename V, typename S, typename O, typename C, typename R>
    struct EventSetupInit<AssociatedVariableCollectionSelector<I, V, S, O, C, R> > {
      typedef AssociatedVariableCollectionSelectorEventSetupInit<AssociatedVariableCollectionSelector<I, V, S, O, C, R> >
          type;
    };

  }  // namespace modules
}  // namespace reco

#endif
