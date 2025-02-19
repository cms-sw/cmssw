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
#include "boost/type_traits/remove_const.hpp"
// user include files

// forward declarations
namespace edm {
   namespace eventsetup {
      namespace heterocontainer {
         template<class Type, class Key, class IdTag>
         Key makeKey(const IdTag&);
   
         template<class Type, class Key>
            Key makeKey();

         //NOTE: the following functions use this struct to determine
         //  how to find the 'Type' (what is returned from the Storage)
         //  when given only an ItemType (what is stored in Storage). 
         //  This allows the Storage to be composed of proxies instead of
         //  the 'Type' themselves
         template<class Key, class ItemType> struct type_from_itemtype {
            typedef typename boost::remove_const<ItemType>::type Type;
         };
   
         template<class Key, class ItemType, class Storage, class IdTag>
            bool insert(Storage&, ItemType*, const IdTag&);
         
         template<class Key, class ItemType, class Storage>
            bool insert(Storage&, ItemType*);
         
         template<class Key, class ItemType, class Storage, class IdTag>
            ItemType* find(const Storage&, const IdTag&);
         
         template<class Key,class ItemType, class Storage>
            ItemType* find(const Storage&);
      }
   }
}
#endif
