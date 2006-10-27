#ifndef RecoAlgos_SortCollectionSelector_h
#define RecoAlgos_SortCollectionSelector_h
/** \class SortCollectionSelector
 *
 * selects the first N elements based on a sorting algorithm
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.4 $
 *
 * $Id: SortCollectionSelector.h,v 1.4 2006/10/27 07:55:03 llista Exp $
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/UtilAlgos/interface/SelectionAdderTrait.h"
#include <algorithm>
#include <utility>
namespace edm { class Event; }

template<typename C, typename CMP, 
	 typename SC = std::vector<const typename C::value_type *>, 
	 typename A = typename helper::SelectionAdderTrait<SC>::type>
class SortCollectionSelector {
  typedef C collection;
  typedef const typename C::value_type * reference;
  typedef std::pair<reference, size_t> pair;
  typedef SC container;
  typedef typename container::const_iterator const_iterator;

public:
  SortCollectionSelector( const edm::ParameterSet & cfg ) : 
    compare_( CMP() ),
    maxNumber_( cfg.template getParameter<unsigned int>( "maxNumber" ) ) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select( const edm::Handle<C> & c, const edm::Event & ) {
    std::vector<pair> v;
    for( size_t idx = 0; idx < c->size(); ++ idx )
      v.push_back( std::make_pair( & ( * c )[ idx ], idx ) );
    std::sort( v.begin(), v.end(), compare_ );
    selected_.clear();
    for( size_t i = 0; i < maxNumber_ && i < v.size(); ++i )
      A::add( selected_, c, v[ i ].second );
  }
private:
  struct Comparator {
    Comparator( const CMP & cmp ) : cmp_( cmp ) { }
    bool operator()( const pair & t1, const pair & t2 ) const {
      return cmp_( * t1.first, * t2.first );
    } 
    CMP cmp_;
  };
  Comparator compare_;
  unsigned int maxNumber_;
  container selected_;
};

#endif
