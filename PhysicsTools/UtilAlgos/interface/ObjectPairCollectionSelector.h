#ifndef RecoAlgos_ObjectPairCollectionSelector_h
#define RecoAlgos_ObjectPairCollectionSelector_h
/** \class ObjectPairCollectionSelector
 *
 * selects object pairs wose combination satiefies a specific selection
 * for instance, could be based on invariant mass, deltaR , deltaPhi, etc.
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 * $Id: ObjectPairCollectionSelector.h,v 1.3 2006/10/27 10:03:45 llista Exp $
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include <vector>
namespace edm { class Event; }

template<typename C, typename S,
	 typename SC = std::vector<const typename C::value_type *>, 
	 typename A = typename helper::SelectionAdderTrait<SC>::type>
class ObjectPairCollectionSelector {
  typedef C collection;
  typedef const typename C::value_type * reference;
  typedef SC container;
  typedef typename container::const_iterator const_iterator;

public:
  ObjectPairCollectionSelector( const edm::ParameterSet & cfg ) : 
    select_( cfg ) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select( const edm::Handle<C> & c, const edm::Event & ) {
    unsigned int s = c->size();
    std::vector<bool> v( s, false );
    for( unsigned int i = 0; i < s; ++ i )
      for( unsigned int j = i + 1; j < s; ++ j ) {
	if ( select_( (*c)[ i ], (*c)[ j ] ) )
	  v[ i ] = v[ j ] = true;
      }
    selected_.clear();
    for( unsigned int i = 0; i < s; ++i )
      if ( v[ i ] ) A::add( selected_, c, i );
  }
  
private:
  S select_;
  container selected_;
};

#endif
