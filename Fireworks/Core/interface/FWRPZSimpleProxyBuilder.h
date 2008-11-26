#ifndef Fireworks_Core_FWRPZSimpleProxyBuilder_h
#define Fireworks_Core_FWRPZSimpleProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZSimpleProxyBuilder
// 
/**\class FWRPZSimpleProxyBuilder FWRPZSimpleProxyBuilder.h Fireworks/Core/interface/FWRPZSimpleProxyBuilder.h

 Description: Base class for 'simple' proxy builders

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 19 10:40:21 EST 2008
// $Id$
//

// system include files
#include <typeinfo>

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations

class FWRPZSimpleProxyBuilder : public FWRPZDataProxyBuilderBase {
   
public:
   FWRPZSimpleProxyBuilder(const std::type_info&);
   virtual ~FWRPZSimpleProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   static
   std::string typeOfBuilder();
   
   // ---------- member functions ---------------------------
   
private:
   FWRPZSimpleProxyBuilder(const FWRPZSimpleProxyBuilder&); // stop default
   
   const FWRPZSimpleProxyBuilder& operator=(const FWRPZSimpleProxyBuilder&); // stop default
   
   void build();
   
   //called once for each item in collection, the void* points to the 
   // object properly offset in memory
   virtual void build(const void*, unsigned int iIndex, TEveElement& iItemHolder) const = 0;
   
   virtual void itemChangedImp(const FWEventItem*);
   virtual void itemBeingDestroyedImp(const FWEventItem*);
   virtual void modelChangesImp(const FWModelIds&);
   virtual TEveElementList* getRhoPhiProduct() const;
   virtual TEveElementList* getRhoZProduct() const;
   // ---------- member data --------------------------------
   FWEvePtr<TEveElementList> m_containerPtr;
   const std::type_info* m_itemType;
   long m_objectOffset;
   mutable bool m_needsUpdate;
};


#endif
