#ifndef RecoAlgos_SingleElementCollectionRefSelector_h
#define RecoAlgos_SingleElementCollectionRefSelector_h
/** \class SingleElementCollectionRefSelector
 *
 * selects a subset of a collection based
 * on single element selection done via functor
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: SingleElementCollectionRefSelector.h,v 1.2 2013/02/28 00:28:05 wmtan Exp $
 *
 */
#include "CommonTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "CommonTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "DataFormats/Common/interface/View.h"
namespace reco {
  namespace modules {
    template<typename S> struct SingleElementCollectionRefSelectorEventSetupInit;
  }
}
template<typename InputType, typename Selector, 
	 typename OutputCollection = typename ::helper::SelectedOutputCollectionTrait<edm::View<InputType> >::type, 
	 typename StoreContainer = typename ::helper::StoreContainerTrait<OutputCollection>::type,
	 typename RefAdder = typename ::helper::SelectionAdderTrait<edm::View<InputType>, StoreContainer>::type>
struct SingleElementCollectionRefSelector {
  typedef edm::View<InputType> InputCollection;
  typedef InputCollection collection;
  typedef StoreContainer container;
  typedef Selector selector;
  typedef typename container::const_iterator const_iterator;
  SingleElementCollectionRefSelector(const edm::ParameterSet & cfg) : 
    select_(reco::modules::make<Selector>(cfg)) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select(const edm::Handle<InputCollection> & c, const edm::Event &, const edm::EventSetup&) {
    selected_.clear();    
    for(size_t idx = 0; idx < c->size(); ++ idx) {
      if(select_(c->refAt(idx))) addRef_(selected_, c, idx);
    }
  }
private:
  container selected_;
  selector select_;
  RefAdder addRef_;
  friend class reco::modules::SingleElementCollectionRefSelectorEventSetupInit<SingleElementCollectionRefSelector>;
};

#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"

namespace reco {
  namespace modules {
    template<typename S>
    struct SingleElementCollectionRefSelectorEventSetupInit {
      static void init(S & s, const edm::Event & ev, const edm::EventSetup& es) { 
	typedef typename EventSetupInit<typename S::selector>::type ESI;
	ESI::init(s.select_, ev, es);
      }
    };

    template<typename I, typename S, typename O, typename C, typename R>
    struct EventSetupInit<SingleElementCollectionRefSelector<I, S, O, C, R> > {
      typedef SingleElementCollectionRefSelectorEventSetupInit<SingleElementCollectionRefSelector<I, S, O, C, R> > type;
    };
  }
}

#endif

