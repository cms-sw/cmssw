#ifndef Common_OneToManyWithQuality_h
#define Common_OneToManyWithQuality_h
#include "DataFormats/Common/interface/AssociationMapHelpers.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <map>
#include <vector>
#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

namespace edm {
  template<typename CKey, typename CVal, typename Q, typename index = unsigned int>
  class OneToManyWithQuality {
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// reference to "value" collection
    typedef edm::RefProd<CVal> ValRefProd;
    /// internal map associated data
    typedef std::vector<std::pair<index, Q> > map_assoc;

  public:
    /// values reference collection type
    typedef std::vector<std::pair<edm::Ref<CVal>, Q> > val_type;
    /// insert key type
    typedef edm::Ref<CKey> key_type;
    /// insert val type
    typedef std::pair<edm::Ref<CVal>, Q> data_type;
    /// index type
    typedef index index_type;
    /// map type
    typedef std::map<index_type, map_assoc> map_type;
    /// reference set type
    typedef helpers::KeyVal<KeyRefProd, ValRefProd> ref_type;
    /// insert in the map
    static void insert( ref_type & ref, map_type & m,
			const key_type & k, const data_type & v ) {
      const edm::Ref<CVal> & vref = v.first;
      if ( k.isNull() || vref.isNull() )
	throw edm::Exception( edm::errors::InvalidReference )
	  << "can't insert null references in AssociationMap";
      if ( ref.key.isNull() ) {
	ref.key = KeyRefProd( k );
	ref.val = ValRefProd( vref );
      }
      helpers::checkRef( ref.key, k ); helpers::checkRef( ref.val, vref );
      index_type ik = index_type( k.key() ), iv = index_type( vref.key() );
      m[ ik ].push_back( std::make_pair( iv, v.second ) );
    }
    static void insert( ref_type & ref, map_type & m, const key_type & k, const val_type & v ) {
      for( typename val_type::const_iterator i = v.begin(), iEnd = v.end(); i != iEnd; ++i )
      insert( ref, m, k, * i );
    }
    /// return values collection
    static val_type val( const ref_type & ref, const map_assoc & iv ) {
      val_type v;
      for( typename map_assoc::const_iterator idx = iv.begin(), idxEnd = iv.end(); idx != idxEnd; ++idx )
	v.push_back( std::make_pair( edm::Ref<CVal>( ref.val, idx->first ), idx->second ) );
      return v;
    }
    /// size of data_type
    static typename map_type::size_type size( const map_assoc & v ) { return v.size(); }
    /// sort
    static void sort( map_type & m ) { 
      //      using namespace boost::lambda;
      for( typename map_type::iterator i = m.begin(), iEnd = m.end(); i != iEnd; ++i ) {
	map_assoc & v = i->second;
	double std::pair<index, Q>:: * quality = & std::pair<index, Q>::second;
	std::sort( v.begin(), v.end(),  
		   bind( quality, boost::lambda::_2 ) < bind( quality, boost::lambda::_1 ) );
      }
    }
  };
}

#endif
