#ifndef RecoAlgos_ObjectPairCollectionSelector_h
#define RecoAlgos_ObjectPairCollectionSelector_h
/** \class ObjectPairCollectionSelector
 *
 * selects object pairs wose combination satiefies a specific selection
 * for instance, could be based on invariant mass, deltaR , deltaPhi, etc.
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ObjectPairCollectionSelector.h,v 1.1 2006/10/03 10:47:40 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include <vector>
namespace edm { class Event; }

template<typename C, typename S>
struct ObjectPairCollectionSelector {
  typedef C collection;
  typedef std::vector<const typename C::value_type *> container;
  typedef typename container::const_iterator const_iterator;
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
      if ( v[ i ] ) selected_.push_back( & (*c)[ i ] );
  }
  
private:
  S select_;
  container selected_;
};

#endif
