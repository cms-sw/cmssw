#ifndef DataFormats_Common_AssociationMapKeyVal_h
#define DataFormats_Common_AssociationMapKeyVal_h
/*
 *
 * helper classes for AssociationMap
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "FWCore/Utilities/interface/EDMException.h"

#include <utility>

namespace edm {

  class EDProductGetter;

  namespace helpers {
    template<typename K, typename V>
    struct KeyVal {
      typedef K key_type;
      typedef V value_type;
      KeyVal() : key(), val() { }
      KeyVal(const K & k, const V & v) : key(k), val(v) { }
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
      template<typename K_, typename V_>
      KeyVal(K_&& k, V_&& v) : key(std::forward<K_>(k)),val(std::forward<V_>(v)){}

      KeyVal(EDProductGetter const* getter) : key(ProductID(), getter), val(ProductID(), getter) { }
#endif

      K key;
      V val;
    };
    
    template<typename K>
    struct Key {
      typedef K key_type;
      Key() { }
      Key(const K & k) : key(k) { }
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
      template<typename K_>
      Key(K_&& k) : key(std::forward<K_>(k)) { }

      Key(EDProductGetter const* getter) : key(ProductID(), getter) { }
#endif

      K key;
    };
    
    /// throw if r hasn't the same id as rp
    template<typename RP, typename R>
    void checkRef(const RP & rp, const R & r) {
      if (rp.id() != r.id()) {
        Exception::throwThis(edm::errors::InvalidReference, "invalid reference");
      }
    }
  }
}

#endif
