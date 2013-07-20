#ifndef RecoAlgos_ObjectPairCollectionSelector_h
#define RecoAlgos_ObjectPairCollectionSelector_h
/** \class ObjectPairCollectionSelector
 *
 * selects object pairs wose combination satiefies a specific selection
 * for instance, could be based on invariant mass, deltaR , deltaPhi, etc.
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: ObjectPairCollectionSelector.h,v 1.2 2013/02/28 00:34:26 wmtan Exp $
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include <vector>

namespace edm { class Event; }

template<typename InputCollection, typename Selector,
	 typename StoreContainer = std::vector<const typename InputCollection::value_type *>, 
	 typename RefAdder = typename helper::SelectionAdderTrait<InputCollection, StoreContainer>::type>
class ObjectPairCollectionSelector {
public:
  typedef InputCollection collection;
  
private:
  typedef const typename InputCollection::value_type * reference;
  typedef StoreContainer container;
  typedef typename container::const_iterator const_iterator;
  
public:
  ObjectPairCollectionSelector(const edm::ParameterSet & cfg) : 
    select_(reco::modules::make<Selector>(cfg)) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select(const edm::Handle<InputCollection> &c, const edm::Event &, const edm::EventSetup &) {
    unsigned int s = c->size();
    std::vector<bool> v(s, false);
    for(unsigned int i = 0; i < s; ++i)
      for(unsigned int j = i + 1; j < s; ++j) {
	if(select_((*c)[i], (*c)[j]))
	  v[i] = v[j] = true;
      }
    selected_.clear();
    for(unsigned int i = 0; i < s; ++i)
    if (v[i]) 
      addRef_(selected_, c, i);
  }
  
private:
  Selector select_;
  StoreContainer selected_;
  RefAdder addRef_;
};

#endif

