#ifndef UtilAlgos_CollectionFilterTrait_h
#define UtilAlgos_CollectionFilterTrait_h
/* \class CollectionFilterTrait<C, S, N>
 *
 * \author Luca Lista, INFN
 *
 */
#include "CommonTools/UtilAlgos/interface/AnySelector.h"
#include "CommonTools/UtilAlgos/interface/MinNumberSelector.h"

namespace helper {

  template<typename C, typename S, typename N>
  struct CollectionFilter {
    static bool filter( const C & source, const S & select, const N & sizeSelect ) {
      size_t n = 0;
      for( typename C::const_iterator i = source.begin(); i != source.end(); ++ i )
	if ( select( * i ) ) n ++;
      return sizeSelect( n );      
    }
  };

  template<typename C, typename S>
  struct CollectionFilter<C, S, MinNumberSelector> {
    static bool filter( const C& source, const S & select, const MinNumberSelector & sizeSelect ) {
      size_t n = 0;
      for( typename C::const_iterator i = source.begin(); i != source.end(); ++ i ) {
	if ( select( * i ) ) n ++;
	if ( sizeSelect( n ) ) return true;
      }
      return false;
    }
  };

  template<typename C, typename N>
  struct CollectionSizeFilter {
    template<typename S>
    static bool filter( const C & source, const S & , const N & sizeSelect ) {
      return sizeSelect( source.size() );
    }
  };

  template<typename C, typename S, typename N>
  struct CollectionFilterTrait {
    typedef CollectionFilter<C, S, N> type;
  };

  template<typename C, typename N>
  struct CollectionFilterTrait<C, AnySelector, N> {
    typedef CollectionSizeFilter<C, N> type;
  };

}

#endif

