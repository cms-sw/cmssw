#ifndef RecoAlgos_SingleElementCollectionRefSelector_h
#define RecoAlgos_SingleElementCollectionRefSelector_h
/** \class SingleElementCollectionRefSelector
 *
 * selects a subset of a collection based
 * on single element selection done via functor
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: SingleElementCollectionRefSelector.h,v 1.1 2009/03/03 13:07:28 llista Exp $
 *
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
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
  SingleElementCollectionRefSelector(const edm::ParameterSet & cfg, edm::ConsumesCollector && iC) :
    select_(reco::modules::make<Selector>(cfg, iC)) { }
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

