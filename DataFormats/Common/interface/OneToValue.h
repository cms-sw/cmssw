#ifndef DataFormats_Common_OneToValue_h
#define DataFormats_Common_OneToValue_h
#include "DataFormats/Common/interface/AssociationMapHelpers.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <map>

namespace edm {
  template<typename CKey, typename Val, typename index = unsigned int>
  class OneToValue {
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// internal map associated data
    typedef Val map_assoc;
  public:
    /// values reference collection type
    typedef Val val_type;
    /// insert key type
    typedef edm::Ref<CKey> key_type;
    /// insert val type
    typedef Val data_type;
    /// index type
    typedef index index_type;
    /// map type
    typedef std::map<index_type, map_assoc> map_type;
    /// reference set type
    typedef helpers::Key<KeyRefProd> ref_type;
    /// transient map type
    typedef std::map<const typename CKey::value_type *, Val> transient_map_type;
    /// transient key vector
    typedef std::vector<const typename CKey::value_type *> transient_key_vector;
    /// transient val vector
    typedef std::vector<Val> transient_val_vector;
    /// insert in the map
    static void insert(ref_type & ref, map_type & m,
			const key_type & k, const data_type & v) {
      if (k.isNull())
	Exception::throwThis(errors::InvalidReference,
	  "can't insert null references in AssociationMap");
      if (ref.key.isNull()) {
	ref.key = KeyRefProd(k);
      }
      helpers::checkRef(ref.key, k);
      index_type ik = index_type(k.key());
      m[ik] = v;
    }
    /// return values collection
    static val_type val(const ref_type & ref, const map_assoc & v) {
      return v;
    }
    /// size of data_type
    static typename map_type::size_type size(const map_assoc &) { return 1; }
    /// sort
    static void sort(map_type &) { }
    /// fill transient map
    static transient_map_type transientMap(const ref_type & ref, const map_type & map) {
      transient_map_type m;
      const CKey & ckey = * ref.key;
      for(typename map_type::const_iterator i = map.begin(); i != map.end(); ++ i) {
	const typename CKey::value_type * k = & ckey[i->first];
	m.insert(std::make_pair(k, i->second));
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
      for(typename map_type::const_iterator i = map.begin(); i != map.end(); ++ i)
	m.push_back(i->second);
      return m;
    }
  };
}

#endif
