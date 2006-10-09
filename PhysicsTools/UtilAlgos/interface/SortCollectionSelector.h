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
 * $Id: SortCollectionSelector.h,v 1.2 2006/07/31 13:30:48 llista Exp $
 *
 */

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
    maxNumber_( cfg.template getParameter<unsigned int>( "maxNumber" ) ) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select( const C & c, const edm::Event & ) {
    container v;
    for( typename C::const_iterator i = c.begin(); i != c.end(); ++ i )
      v.push_back( & * i );
    std::sort( v.begin(), v.end(), compare_ );
    selected_.clear();
    for( unsigned int i = 0; i < maxNumber_ && i < v.size(); ++i )
      selected_.push_back( v[ i ] );
  }
private:
  struct Comparator {
    typedef typename C::value_type value_type;
    Comparator( const CMP & cmp ) : cmp_( cmp ) { }
    bool operator()( const value_type * t1, const value_type * t2 ) const {
      return cmp_( * t1, * t2 );
    } 
    CMP cmp_;
  };
  Comparator compare_;
  unsigned int maxNumber_;
  container selected_;
};

#endif
