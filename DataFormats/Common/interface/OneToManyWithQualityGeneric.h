#ifndef DataFormats_Common_OneToManyWithQualityGeneric_h
#define DataFormats_Common_OneToManyWithQualityGeneric_h
#include "DataFormats/Common/interface/AssociationMapHelpers.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <functional>
#include <map>
#include <vector>
#include <algorithm>
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
    static void insert(ref_type& ref, map_type & m,
			const key_type & k, const data_type & v) {
      const ValRef & vref = v.first;
      if (k.isNull() || vref.isNull())
	Exception::throwThis(errors::InvalidReference,
	  "can't insert null references in AssociationMap");
      if(ref.key.isNull()) {
        if(k.isTransient() || vref.isTransient()) {
          Exception::throwThis(errors::InvalidReference,
	    "can't insert transient references in uninitialized AssociationMap");
        }
        //another thread might cause productGetter() to return a different value
        EDProductGetter const* getter = ref.key.productGetter();
        if(getter == nullptr) {
          Exception::throwThis(errors::LogicError,
            "Can't insert into AssociationMap unless it was properly initialized.\n"
            "The most common fix for this is to add arguments to the call to the\n"
            "AssociationMap constructor that are valid Handle's to the containers.\n"
            "If you don't have valid handles or either template parameter to the\n"
            "AssociationMap is a View, then see the comments in AssociationMap.h.\n"
            "(note this was a new requirement added in the 7_5_X release series)\n");
        }
        ref.key = KeyRefProd(k.id(), getter);
        ref.val = ValRefProd(vref.id(), ref.val.productGetter());
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
    // Note the Framework automatically calls this after putting the object
    // into the event using AssociationMap::post_insert. It sorts in reverse
    // order of the quality.
    static void sort(map_type & m) {
      //      using namespace boost::lambda;
      for(typename map_type::iterator i = m.begin(), iEnd = m.end(); i != iEnd; ++i) {
        using std::placeholders::_1;
        using std::placeholders::_2;
	map_assoc & v = i->second;
	// Q std::pair<index, Q>::*quality = &std::pair<index, Q>::second;
	// std::sort(v.begin(), v.end(),
	// 	  bind(quality, boost::lambda::_2) < bind(quality, boost::lambda::_1));
           std::sort(v.begin(), v.end(), 
                  std::bind(std::less<Q>(), 
                  std::bind(&std::pair<index, Q>::second,_2), std::bind( &std::pair<index, Q>::second,_1)
                             )
           );

      }
    }
    /// fill transient map
    static transient_map_type transientMap(const ref_type & ref, const map_type & map) {
      transient_map_type m;
      if(!map.empty()) {
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
      }
      return m;
    }
    /// fill transient key vector
    static transient_key_vector transientKeyVector(const ref_type & ref, const map_type & map) {
      transient_key_vector m;
      if(!map.empty()) {
        const CKey & ckey = * ref.key;
        for(typename map_type::const_iterator i = map.begin(); i != map.end(); ++ i)
          m.push_back(& ckey[i->first]);
      }
      return m;
    }
    /// fill transient val vector
    static transient_val_vector transientValVector(const ref_type & ref, const map_type & map) {
      transient_val_vector m;
      if(!map.empty()) {
        const CVal & cval = * ref.val;
        for(typename map_type::const_iterator i = map.begin(); i != map.end(); ++ i) {
          const map_assoc & a = i->second;
          std::vector<std::pair<const typename CVal::value_type *, Q> > v;
          m.push_back(v);
          for(typename map_assoc::const_iterator j = a.begin(); j != a.end(); ++j)
            m.back().push_back(std::make_pair(& cval[ j->first ], j->second));
        }
      }
      return m;
    }
  };
}

#endif
