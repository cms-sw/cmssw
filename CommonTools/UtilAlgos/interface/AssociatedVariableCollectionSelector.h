#ifndef CommonTools_UtilAlgos_AssociatedVariableCollectionSelector_h
#define CommonTools_UtilAlgos_AssociatedVariableCollectionSelector_h
/* \class AssociatedVariableCollectionSelector
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: AssociatedVariableCollectionSelector.h,v 1.3 2013/02/28 00:34:26 wmtan Exp $
 *
 */
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CommonTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "CommonTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "DataFormats/Common/interface/getRef.h"

namespace reco {
  namespace modules {
    template<typename S> struct AssociatedVariableCollectionSelectorEventSetupInit;
  }
}

template<typename InputCollection, typename VarCollection, typename Selector,
	 typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<InputCollection>::type, 
	 typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
	 typename RefAdder = typename ::helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
class AssociatedVariableCollectionSelector {
public:
  typedef InputCollection collection;
  typedef StoreContainer container;
  typedef Selector selector;
  typedef typename container::const_iterator const_iterator;
  AssociatedVariableCollectionSelector(const edm::ParameterSet & cfg) : 
    var_(cfg.template getParameter<edm::InputTag>("var")),
    select_(reco::modules::make<Selector>(cfg)) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select(const edm::Handle<InputCollection>& c, const edm::Event& evt, const edm::EventSetup&) {
    selected_.clear();    
    edm::Handle<VarCollection> var;
    evt.getByLabel(var_, var);
    for(size_t idx = 0; idx < c->size(); ++idx) {
      if (select_((*c)[idx], (*var)[edm::getRef(c,idx)])) 
	addRef_(selected_, c, idx);
    }
  }
private:
  edm::InputTag var_;
  container selected_;
  selector select_;
  RefAdder addRef_;
  friend class reco::modules::AssociatedVariableCollectionSelectorEventSetupInit<AssociatedVariableCollectionSelector>;
};


#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"

namespace reco {
  namespace modules {
    template<typename S>
    struct AssociatedVariableCollectionSelectorEventSetupInit {
      static void init(S & s, const edm::Event& evt, const edm::EventSetup& es) { 
	typedef typename EventSetupInit<typename S::selector>::type ESI;
	ESI::init(s.select_, evt, es);
      }
    };

    template<typename I, typename V, typename S, typename O, typename C, typename R>
    struct EventSetupInit<AssociatedVariableCollectionSelector<I, V, S, O, C, R> > {
      typedef AssociatedVariableCollectionSelectorEventSetupInit<AssociatedVariableCollectionSelector<I, V, S, O, C, R> > type;
    };
  }
}

#endif

