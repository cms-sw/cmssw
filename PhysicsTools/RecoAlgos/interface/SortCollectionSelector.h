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
 * $Id: SortCollectionSelector.h,v 1.2 2006/07/21 16:58:23 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <algorithm>
namespace edm { class Event; }

template<typename C, typename CMP>
struct SortCollectionSelector {
  typedef C collection;
  typedef std::vector<const typename C::value_type *> container;
  typedef typename container::const_iterator const_iterator;
  SortCollectionSelector( const edm::ParameterSet & cfg ) : 
    compare_( CMP() ),
    max_( cfg.template getParameter<unsigned int>( "max" ) ) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select( const reco::TrackCollection & c, const edm::Event & ) {
    container v;
    for( typename C::const_iterator i = c.begin(); i != c.end(); ++ i )
      v.push_back( & * i );
    std::sort( v.begin(), v.end(), compare_ );
    selected_.clear();
    for( unsigned int i = 0; i < max_ && i < v.size(); ++i )
      selected_.push_back( v[ i ] );
  }
private:
  struct Comparator {
    Comparator( const CMP & cmp ) : cmp_( cmp ) { }
    bool operator()( const reco::Track * t1, const reco::Track * t2 ) const {
      return cmp_( * t1, * t2 );
    } 
    CMP cmp_;
  };
  Comparator compare_;
  unsigned int max_;
  container selected_;
};

#endif
