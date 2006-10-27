#ifndef RecoAlgos_SingleElementRefVectorCollectionSelector_h
#define RecoAlgos_SingleElementRefVectorCollectionSelector_h
/** \class SingleElementCollectionSelector
 *
 * selects a subset of a track collection based
 * on single element selection done via functor
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: SingleElementCollectionSelector.h,v 1.2 2006/07/25 17:20:27 llista Exp $
 *
 */
#include <vector>
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace edm { class Event; }

template<typename C, typename S>
struct SingleElementRefVectorCollectionSelector {
  typedef C collection;
  typedef edm::RefVector<C> container;
  typedef typename container::const_iterator const_iterator;
  SingleElementRefVectorCollectionSelector( const edm::ParameterSet & cfg ) : 
    select_( cfg ) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select( const edm::Handle<C> & c, const edm::Event & ) {
    selected_.clear();
    size_t idx = 0;
    for( typename C::const_iterator i = c->begin(); i != c->end(); ++ i, ++ idx )
      if ( select_( * i ) ) selected_.push_back( edm::Ref<C>( c, idx ) );
  }
private:
  container selected_;
  S select_;
};

#endif
