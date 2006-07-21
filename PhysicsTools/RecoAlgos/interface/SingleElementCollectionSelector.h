#ifndef RecoAlgos_SingleElementCollectionSelector_h
#define RecoAlgos_SingleElementCollectionSelector_h
/** \class SingleElementCollectionSelector
 *
 * selects a subset of a track collection based
 * on single element selection done via functor
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: SingleElementCollectionSelector.h,v 1.1 2006/07/21 14:11:26 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include <vector>

template<typename C, typename S>
struct SingleElementCollectionSelector {
  typedef std::vector<const typename C::value_type *> container;
  typedef typename container::const_iterator const_iterator;
  SingleElementCollectionSelector( const edm::ParameterSet & cfg ) : 
    select_( cfg ) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select( const reco::TrackCollection & c ) {
    selected_.clear();
    for( typename C::const_iterator i = c.begin(); i != c.end(); ++ i )
      if ( select_( * i ) ) selected_.push_back( & * i );
  }
private:
  container selected_;
  S select_;
};

#endif
