#ifndef DataFormats_Common_OneToManyWithQualityGeneric_h
#define DataFormats_Common_OneToManyWithQualityGeneric_h
#include "DataFormats/Common/interface/AssociationMapHelpers.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <map>
#include <vector>
#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include "DataFormats/Common/interface/MapRefViewTrait.h"

namespace edm {
  template<typename CKey, typename CVal, typename Q, typename index = unsigned int,
    typename KeyRefProd = typename helper::MapRefViewTrait<CKey>::refprod_type, 
    typename ValRefProd = typename helper::MapRefViewTrait<CVal>::refprod_type, 
    typename KeyRef = typename helper::MapRefViewTrait<CKey>::ref_type, 
    typename ValRef = typename helper::MapRefViewTrait<CVal>::ref_type >
  class OneToManyWithQualityGeneric {
    /// reference to "key" collection
    typedef KeyRefProd keyrefprod_type;
    /// reference to "value" collection
    typedef ValRefProd valrefprod_type;    
    /// internal map associated data
    typedef std::vector<std::pair<index, Q> > map_assoc;

  public:
    /// values reference collection type
    typedef std::vector<std::pair<ValRef, Q> > val_type;
    /// insert key type
    typedef KeyRef key_type;
    /// insert val type
    typedef std::pair<ValRef, Q> data_type;
    /// index type
    typedef index index_type;
    /// map type
    typedef std::map<index_type, map_assoc> map_type;
    /// reference set type
    typedef helpers::KeyVal<keyrefprod_type, valrefprod_type> ref_type;
    /// transient map type
    typedef std::map<const typename CKey::value_type *,
    		     std::vector<std::pair<const typename CVal::value_type *, Q > > 
                    > transient_map_type;
    /// transient key vector
    typedef std::vector<const typename CKey::value_type *> transient_key_vector;
    /// transient val vector
    typedef std::vector<std::vector<std::pair<const typename CVal::value_type *, Q > > 
                       > transient_val_vector;
    /// insert in the map
    static void insert(ref_type & ref, map_type & m,
			const key_type & k, const data_type & v) {
      const ValRef & vref = v.first;
      if (k.isNull() || vref.isNull())
	throw edm::Exception(edm::errors::InvalidReference)
	  << "can't insert null references in AssociationMap";
      if (ref.key.isNull()) {
	ref.key = keyrefprod_type(k);
	ref.val = valrefprod_type(vref);
      }
      helpers::checkRef(ref.key, k); helpers::checkRef(ref.val, vref);
      index_type ik = index_type(k.key()), iv = index_type(vref.key());
      m[ik].push_back(std::make_pair(iv, v.second));
    }
    static void insert(ref_type & ref, map_type & m, const key_type & k, const val_type & v) {
      for(typename val_type::const_iterator i = v.begin(), iEnd = v.end(); i != iEnd; ++i)
      insert(ref, m, k, *i);
    }
    /// return values collection
    static val_type val(const ref_type & ref, const map_assoc & iv) {
      val_type v;
      for(typename map_assoc::const_iterator idx = iv.begin(), idxEnd = iv.end(); idx != idxEnd; ++idx)
	v.push_back(std::make_pair(ValRef(ref.val, idx->first), idx->second));
      return v;
    }
    /// size of data_type
    static typename map_type::size_type size(const map_assoc & v) { return v.size(); }
    /// sort
    static void sort(map_type & m) {
      //      using namespace boost::lambda;
      for(typename map_type::iterator i = m.begin(), iEnd = m.end(); i != iEnd; ++i) {
	map_assoc & v = i->second;
	Q std::pair<index, Q>::*quality = &std::pair<index, Q>::second;
	std::sort(v.begin(), v.end(),
		  bind(quality, boost::lambda::_2) < bind(quality, boost::lambda::_1));
      }
    }
    /// fill transient map
    static transient_map_type transientMap(const ref_type & ref, const map_type & map) {
      transient_map_type m;
      const CKey & ckey = * ref.key;
      const CVal & cval = * ref.val;
      for(typename map_type::const_iterator i = map.begin(); i != map.end(); ++ i) {
	const map_assoc & a = i->second;
	const typename CKey::value_type * k = & ckey[i->first];
	std::vector<std::pair<const typename CVal::value_type *, Q> > v;
	for(typename map_assoc::const_iterator j = a.begin(); j != a.end(); ++j) {
	  const typename CVal::value_type * val = & cval[j->first];
	  v.push_back(std::make_pair(val, j->second));
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
	std::vector<std::pair<const typename CVal::value_type *, Q> > v;
	m.push_back(v);
	for(typename map_assoc::const_iterator j = a.begin(); j != a.end(); ++j)
	  m.back().push_back(std::make_pair(& cval[ j->first ], j->second));
      }
      return m;
    }
  };
}

#endif
