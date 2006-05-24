#ifndef Common_KeyVal_h
#define Common_KeyVal_h
/** \class edm::KeyVal
 *
 * key-value pair.
 * 
 * \author Luca Lista, INFN
 *
 * $Id: ValueMap.h,v 1.5 2006/05/23 11:01:07 llista Exp $
 *
 */

namespace edm {
  template<typename K, typename V>
  struct KeyVal {
    KeyVal() { }
    KeyVal( const K & k, const V & v ) : key( k ), val( v ) { }
    K key;
    V val;
  };
}

#endif
