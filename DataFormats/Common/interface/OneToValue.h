#ifndef Common_OneToValue_h
#define Common_OneToValue_h
#include "DataFormats/Common/interface/AssociationMapHelpers.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <map>

namespace edm {
  template<typename CKey, typename Val, typename index = unsigned int>
  class OneToValue {
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// values reference collection type
    typedef Val val_type;
    /// internal map associated data
    typedef Val map_assoc;
  public:
    /// insert key type
    typedef edm::Ref<CKey> key_type;
    /// insert val type
    typedef Val data_type;
    /// index type
    typedef index index_type;
    /// map type
    typedef std::map<index_type, map_assoc> map_type;
    /// value type
    typedef helpers::KeyVal<key_type, val_type> value_type;
    /// reference set type
    typedef helpers::Key<KeyRefProd> ref_type;
    /// insert in the map
    static void insert( ref_type & ref, map_type & m,
			const key_type & k, const data_type & v ) {
      if ( k.isNull() )
	throw edm::Exception( edm::errors::InvalidReference )
	  << "can't insert null references in AssociationMap";
      if ( ref.key.isNull() ) {
	ref.key = KeyRefProd( k );
      }
      helpers::checkRef( ref.key, k );
      index_type ik = index_type( k.key() );
      m[ ik ] = v;
    }
    /// return values collection
    static val_type val( const ref_type & ref, const map_assoc & v ) {
      return v;
    }
    /// size of data_type
    static typename map_type::size_type size( const map_assoc & ) { return 1; }
    /// sort
    static void sort( map_type & ) { }
  };
}

#endif
