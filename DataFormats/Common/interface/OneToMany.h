#ifndef Common_OneToMany_h
#define Common_OneToMany_h
#include "DataFormats/Common/interface/AssociationMapHelpers.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <map>
#include <vector>

namespace edm {
  template<typename CKey, typename CVal, typename index = unsigned int>
  class OneToMany {
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// reference to "value" collection
    typedef edm::RefProd<CVal> ValRefProd;
    /// internal map associated data
    typedef std::vector<index> map_assoc;
  public:
    /// values reference collection type
    typedef edm::RefVector<CVal> val_type;
    /// insert key type
    typedef edm::Ref<CKey> key_type;
    /// insert val type
    typedef edm::Ref<CVal> data_type;
    /// index type
    typedef index index_type;
    /// map type
    typedef std::map<index_type, map_assoc > map_type;
    /// reference set type
    typedef helpers::KeyVal<KeyRefProd, ValRefProd> ref_type;
    /// insert in the map
    static void insert( ref_type & ref, map_type & m,
			const key_type & k, const data_type & v ) {
      if ( k.isNull() || v.isNull() )
	throw edm::Exception( edm::errors::InvalidReference )
	  << "can't insert null references in AssociationMap";
      if ( ref.key.isNull() ) {
	ref.key = KeyRefProd( k );
	ref.val = ValRefProd( v );
      }
      helpers::checkRef( ref.key, k ); helpers::checkRef( ref.val, v );
      index_type ik = index_type( k.key() ), iv = index_type( v.key() );
      m[ ik ].push_back( iv );
    }
    static void insert( ref_type & ref, map_type & m, const key_type & k, const val_type & v ) {
      for( typename val_type::const_iterator i = v.begin(), iEnd = v.end(); i != iEnd; ++i )
      insert( ref, m, k, * i );
    }
    /// return values collection
    static val_type val( const ref_type & ref, const map_assoc & iv ) {
      val_type v;
      for( typename map_assoc::const_iterator idx = iv.begin(), idxEnd = iv.end(); idx != idxEnd; ++idx )
	v.push_back( edm::Ref<CVal>( ref.val, * idx ) );
      return v;
    }
    /// size of data_type
    static typename map_type::size_type size( const map_assoc & v ) { return v.size(); }
    /// sort
    static void sort( map_type & ) { }
  };
}

#endif
