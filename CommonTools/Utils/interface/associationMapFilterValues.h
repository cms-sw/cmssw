#ifndef CommonTools_Utils_associationMapFilterValues_h
#define CommonTools_Utils_associationMapFilterValues_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/AssociationMapHelpers.h"

#include <unordered_set>

namespace associationMapFilterValuesHelpers {
  // Common implementation
  template <typename T_AssociationMap, typename T_Key, 
            typename T_ValueIndex, typename T_Value,
            typename T_ValueIndices>
  void findInsert(T_AssociationMap& ret, const T_Key& key,
                  const T_ValueIndex& valueIndex, const T_Value& value,
                  const T_ValueIndices& value_indices) {
    if(value_indices.find(valueIndex) != value_indices.end()) {
      ret.insert(key, value);
    }
  }

  // By default no implementation, as it turns out to be very specific for the types
  template <typename ValueTag>
  struct IfFound;

  // Specialize for Ref and RefToBase, implementation is the same
  template <typename C, typename T, typename F>
  struct IfFound<edm::Ref<C, T, F>> {
    template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
    static void insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
      findInsert(ret, keyValue.key, keyValue.val.key(), keyValue.val, value_indices);
    }
  };

  template <typename T>
  struct IfFound<edm::RefToBase<T>> {
    template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
    static void insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
      findInsert(ret, keyValue.key, keyValue.val.key(), keyValue.val, value_indices);
    }
  };

  // Specialize for RefVector
  template <typename C, typename T, typename F>
  struct IfFound<edm::RefVector<C, T, F>> {
    template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
    static void insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
      for(const auto& value: keyValue.val) {
        findInsert(ret, keyValue.key, value.key(), value, value_indices);
      }
    }
  };

  // Specialize for vector<pair<Ref, Q>> for OneToManyWithQuality
  template <typename C, typename T, typename F, typename Q>
  struct IfFound<std::vector<std::pair<edm::Ref<C, T, F>, Q> > > {
    template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
    static void insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
      for(const auto& value: keyValue.val) {
        findInsert(ret, keyValue.key, value.first.key(), value, value_indices);
      }
    }
  };

  // Specialize for vector<pair<RefToBase, Q>> for OneToManyWithQuality
  template <typename T, typename Q>
  struct IfFound<std::vector<std::pair<edm::RefToBase<T>, Q> > > {
    template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
    static void insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
      for(const auto& value: keyValue.val) {
        findInsert(ret, keyValue.key, value.first.key(), value, value_indices);
      }
    }
  };

  // Default implementation for RefVector or vector<Ref>
  template <typename T_RefVector>
  struct FillIndices {
    template <typename T_Set, typename T_RefProd>
    static
    void fill(T_Set& set, const T_RefVector& valueRefs, const T_RefProd& refProd) {
      for(const auto& ref: valueRefs) {
        edm::helpers::checkRef(refProd.val, ref);
        set.insert(ref.key());
      }
    }
  };

  // Specialize for View
  template <typename T>
  struct FillIndices<edm::View<T> > {
    template <typename T_Set, typename T_RefProd>
    static
    void fill(T_Set& set, const edm::View<T>& valueView, const T_RefProd& refProd) {
      for(size_t i=0; i<valueView.size(); ++i) {
        const auto& ref = valueView.refAt(i);
        edm::helpers::checkRef(refProd.val, ref);
        set.insert(ref.key());
      }
    }
  };
}

/**
 * Filters entries of AssociationMap by keeping only those
 * associations that have a value in a given collection
 *
 * @param map        AssociationMap to filter
 * @param valueRefs  Collection of Refs to values.
 *
 * @tparam T_AssociationMap  Type of the AssociationMap
 * @tparam T_RefVector       Type of the Ref collection.
 *
 * For AssociationMap<Tag<CKey, CVal>>, the collection of Refs can be
 * RefVector<CVal>, vector<Ref<CVal>>, vector<RefToBase<CVal>>, or
 * View<T>. More can be supported if needed.
 *
 * @return A filtered copy of the AssociationMap
 *
 * Throws if the values of AssociationMap and valueRefs point to
 * different collections (similar check as in
 * AssociationMap::operator[] for the keys).
 */
template <typename T_AssociationMap, typename T_RefVector>
T_AssociationMap associationMapFilterValues(const T_AssociationMap& map, const T_RefVector& valueRefs) {
  // If the input map is empty, just return it in order to avoid an
  // exception from failing edm::helpers::checkRef() (in this case the
  // refProd points to (0,0) that will fail the check).
  if(map.empty())
    return map;

  T_AssociationMap ret(map.refProd());

  // First copy the keys of values to a set for faster lookup of their existence in the map
  std::unordered_set<typename T_AssociationMap::index_type> value_indices;
  associationMapFilterValuesHelpers::FillIndices<T_RefVector>::fill(value_indices, valueRefs, map.refProd());

  for(const auto& keyValue: map) {
    associationMapFilterValuesHelpers::IfFound<typename T_AssociationMap::value_type::value_type>::insert(ret, keyValue, value_indices);
  }
    

  return ret;
}

#endif

