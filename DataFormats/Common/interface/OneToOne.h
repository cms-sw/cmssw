#ifndef Common_OneToOne_h
#define Common_OneToOne_h
#include "DataFormats/Common/interface/AssociationMapHelpers.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <map>

namespace edm {
  template<typename CKey, typename CVal, typename index = unsigned int>
  class OneToOne {
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// reference to "value" collection
    typedef edm::RefProd<CVal> ValRefProd;
    /// internal map associated data
    typedef index map_assoc;

  public:
    /// values reference collection type
    typedef edm::Ref<CVal> val_type;
    /// insert key type
    typedef edm::Ref<CKey> key_type;
    /// insert val type
    typedef edm::Ref<CVal> data_type;
    /// index type
    typedef index index_type;
    /// map type
    typedef std::map<index_type, map_assoc> map_type;
    /// reference set type
    typedef helpers::KeyVal<KeyRefProd, ValRefProd> ref_type;
    /// transient map type
    typedef std::map<const typename CKey::value_type *,
    		     const typename CVal::value_type *> transient_map_type;
    /// insert in the map
    static void insert(ref_type & ref, map_type & m,
		       const key_type & k, const data_type & v) {
      if (k.isNull() || v.isNull())
	throw edm::Exception(edm::errors::InvalidReference)
	  << "can't insert null references in AssociationMap";
      if (ref.key.isNull()) {
	ref.key = KeyRefProd(k);
	ref.val = ValRefProd(v);
      }
      helpers::checkRef(ref.key, k); helpers::checkRef(ref.val, v);
      index_type ik = index_type(k.key()), iv = index_type(v.key());
      m[ik] = iv;
    }
    /// return values collection
    static val_type val(const ref_type & ref, map_assoc iv) {
      return val_type(ref.val, iv);
    }
    /// size of data_type
    static typename map_type::size_type size(const map_assoc & v) { return 1; }
    /// sort
    static void sort(map_type &) { }
    /// fill transient map
    static transient_map_type transientMap(const ref_type & ref, const map_type & map) {
      transient_map_type m;
      const CKey & ckey = * ref.key;
      const CVal & cval = * ref.val;
      for(typename map_type::const_iterator i = map.begin(); i != map.end(); ++ i) {
	const typename CKey::value_type * k = & ckey[i->first];
	const typename CVal::value_type * v = & cval[i->second];
	m.insert(std::make_pair(k, v));
      }
      return m;
    }
  };
}

#endif
