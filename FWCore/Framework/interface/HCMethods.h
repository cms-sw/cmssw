#ifndef Framework_HCMethods_h
#define Framework_HCMethods_h
// -*- C++ -*-
//
// Package:     HeteroContainer
// Module:      HCMethods
// 
// Description: Templated methods to be used to 'construct' a 
//              heterogenous container
//
// Usage:
//    <usage>
//
// Author:      Chris D. Jones
// Created:     Sun Mar 27 15:58:05 EDT 2005
//

// system include files

#include "FWCore/Framework/interface/HCTypeTag.h"
#include <type_traits>
// user include files

namespace edm {
  namespace eventsetup {
    namespace heterocontainer {
      template< class Type, class Key, class IdTag >
      inline Key makeKey(const IdTag& iIdTag) {
	HCTypeTag typeTag = HCTypeTag::make<Type>();
	return Key(typeTag, iIdTag);
      }
      
      template< class Type, class Key>
      inline Key makeKey() {
	HCTypeTag typeTag = HCTypeTag::make<Type>();
	return Key(typeTag);
      }
      
      //NOTE: the following functions use this struct to determine
      //  how to find the 'Type' (what is returned from the Storage)
      //  when given only an ItemType (what is stored in Storage). 
      //  This allows the Storage to be composed of proxies instead of
      //  the 'Type' themselves
      template<class Key, class ItemType> struct type_from_itemtype {
	typedef typename std::remove_const<ItemType>::type Type;
      };

      template<class Key, class ItemType, class Storage, class IdTag >
      inline bool insert(Storage& iStorage, ItemType* iItem, const IdTag& iIdTag) {
	return iStorage.insert(makeKey< typename type_from_itemtype<Key, ItemType>::Type, 
			       Key>(iIdTag) , iItem);
      }
      
      template<  class Key, class ItemType, class Storage>
      inline bool insert(Storage& iStorage, ItemType* iItem) {
	return iStorage.insert(makeKey<ItemType,
			       Key>() , iItem);
         }
      
      
      template< class Key, class ItemType, class Storage, class IdTag >
      inline ItemType* find(const Storage& iStorage, const IdTag& iIdTag) {
	//The cast should be safe since the Key tells us the type
	return static_cast<ItemType*>(iStorage.find(
						    makeKey<typename type_from_itemtype<Key,
						    ItemType>::Type,Key>(iIdTag)));
      }
      
      template< class Key, class ItemType, class Storage>
      inline ItemType* find(const Storage& iStorage) {
	//The cast should be safe since the Key tells us the type
	return static_cast<ItemType*>( iStorage.find(
						     makeKey<typename type_from_itemtype<Key,
						     ItemType>::Type,Key>()));
         }
    }
  }
}
#endif
