#ifndef DataFormats_Common_OneToMany_h
#define DataFormats_Common_OneToMany_h
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
    /// transient map type
    typedef std::map<const typename CKey::value_type *,
    		     std::vector<const typename CVal::value_type *> 
                    > transient_map_type;
    /// transient key vector
    typedef std::vector<const typename CKey::value_type *> transient_key_vector;
    /// transient val vector
    typedef std::vector<std::vector<const typename CVal::value_type *> > transient_val_vector;
    /// insert in the map
    static void insert(ref_type & ref, map_type & m,
			const key_type & k, const data_type & v) {
      if (k.isNull() || v.isNull())
	Exception::throwThis(errors::InvalidReference,
	  "can't insert null references in AssociationMap");
      if (ref.key.isNull()) {
	ref.key = KeyRefProd(k);
	ref.val = ValRefProd(v);
      }
      helpers::checkRef(ref.key, k); helpers::checkRef(ref.val, v);
      index_type ik = index_type(k.key()), iv = index_type(v.key());
      m[ ik ].push_back(iv);
    }
    static void insert(ref_type & ref, map_type & m, const key_type & k, const val_type & v) {
      for(typename val_type::const_iterator i = v.begin(), iEnd = v.end(); i != iEnd; ++i)
      insert(ref, m, k, *i);
    }
    /// return values collection
    static val_type val(const ref_type & ref, const map_assoc & iv) {
      val_type v;
      for(typename map_assoc::const_iterator idx = iv.begin(), idxEnd = iv.end(); idx != idxEnd; ++idx)
	v.push_back(edm::Ref<CVal>(ref.val, *idx));
      return v;
    }
    /// size of data_type
    static typename map_type::size_type size(const map_assoc & v) { return v.size(); }
    /// sort
    static void sort(map_type &) { }
    /// fill transient map
    static transient_map_type transientMap(const ref_type & ref, const map_type & map) {
      transient_map_type m;
      const CKey & ckey = * ref.key;
      const CVal & cval = * ref.val;
      for(typename map_type::const_iterator i = map.begin(); i != map.end(); ++ i) {
	const map_assoc & a = i->second;
	const typename CKey::value_type * k = & ckey[ i->first ];
	std::vector<const typename CVal::value_type *> v;
	for(typename map_assoc::const_iterator j = a.begin(); j != a.end(); ++j) {
	  const typename CVal::value_type * val = & cval[ *j ];
	  v.push_back(val);
	}
	m.insert(std::make_pair(k, v));
      }
      return m;
    }
    /// fill transient key vector
    static transient_key_vector transientKeyVector(const ref_type & ref, const map_type & map) {
      transient_key_vector m;
      const CKey & ckey = * ref.key;
      for(typename map_type::const_iterator i = map.begin(); i != map.end(); ++ i)
	m.push_back(& ckey[i->first]);
      return m;
    }
    /// fill transient val vector
    static transient_val_vector transientValVector(const ref_type & ref, const map_type & map) {
      transient_val_vector m;
      const CVal & cval = * ref.val;
      for(typename map_type::const_iterator i = map.begin(); i != map.end(); ++ i) {
	const map_assoc & a = i->second;
	std::vector<const typename CVal::value_type *> v;
	m.push_back(v);
	for(typename map_assoc::const_iterator j = a.begin(); j != a.end(); ++j)
	  m.back().push_back(& cval[ *j ]);
      }
      return m;
    }
  };
}

#endif
