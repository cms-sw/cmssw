#ifndef RecoAlgos_SortCollectionSelector_h
#define RecoAlgos_SortCollectionSelector_h
/** \class SortCollectionSelector
 *
 * selects the first N elements based on a sorting algorithm
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: SortCollectionSelector.h,v 1.2 2013/02/28 00:28:06 wmtan Exp $
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "CommonTools/UtilAlgos/interface/StoreContainerTrait.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include <algorithm>
#include <utility>
namespace edm { class Event; }

template<typename InputCollection, typename Comparator, 
	 typename OutputCollection = typename helper::SelectedOutputCollectionTrait<InputCollection>::type, 
	 typename StoreContainer = typename helper::StoreContainerTrait<OutputCollection>::type,
	 typename RefAdder = typename helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
class SortCollectionSelector {
public:
  typedef InputCollection collection;
private:
  typedef const typename InputCollection::value_type * reference;
  typedef std::pair<reference, size_t> pair;
  typedef StoreContainer container;
  typedef typename container::const_iterator const_iterator;

public:
  SortCollectionSelector(const edm::ParameterSet & cfg) : 
    compare_(Comparator()),
    maxNumber_(cfg.template getParameter<unsigned int>("maxNumber")) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select(const edm::Handle<InputCollection> & c, const edm::Event &, const edm::EventSetup&) {
    std::vector<pair> v;
    for(size_t idx = 0; idx < c->size(); ++ idx)
      v.push_back(std::make_pair(&(*c)[idx], idx));
    std::sort(v.begin(), v.end(), compare_);
    selected_.clear();
    for(size_t i = 0; i < maxNumber_ && i < v.size(); ++i)
      addRef_(selected_, c, v[i].second);
  }
private:
  struct PairComparator {
    PairComparator(const Comparator & cmp) : cmp_(cmp) { }
    bool operator()(const pair & t1, const pair & t2) const {
      return cmp_(*t1.first, *t2.first);
    } 
    Comparator cmp_;
  };
  PairComparator compare_;
  unsigned int maxNumber_;
  StoreContainer selected_;
  RefAdder addRef_;
};

#endif

