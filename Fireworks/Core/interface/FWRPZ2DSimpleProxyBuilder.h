#ifndef Fireworks_Core_FWRPZ2DSimpleProxyBuilder_h
#define Fireworks_Core_FWRPZ2DSimpleProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZ2DSimpleProxyBuilder
// 
/**\class FWRPZ2DSimpleProxyBuilder FWRPZ2DSimpleProxyBuilder.h Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 26 11:02:04 EST 2008
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBaseFactory.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
class FWRPSimpleCaller;
class FWRZSimpleCaller;

class FWRPZ2DSimpleProxyBuilder : public FWRPZDataProxyBuilderBase {

public:
   friend class FWRPSimpleCaller;
   friend class FWRZSimpleCaller;
   
   FWRPZ2DSimpleProxyBuilder(const std::type_info&);
   virtual ~FWRPZ2DSimpleProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   static
   std::string typeOfBuilder();
   
   // ---------- member functions ---------------------------
   
private:
   FWRPZ2DSimpleProxyBuilder(const FWRPZ2DSimpleProxyBuilder&); // stop default
   
   const FWRPZ2DSimpleProxyBuilder& operator=(const FWRPZ2DSimpleProxyBuilder&); // stop default

   //called once for each item in collection, the void* points to the 
   // object properly offset in memory
   virtual void buildRhoPhi(const void*, unsigned int iIndex, TEveElement& iItemHolder) const = 0;
   virtual void buildRhoZ(const void*, unsigned int iIndex, TEveElement& iItemHolder) const = 0;
   
   //abstract from parent class
   virtual void itemChangedImp(const FWEventItem*) ;
   virtual void itemBeingDestroyedImp(const FWEventItem*);
   virtual void modelChangesImp(const FWModelIds&);
   virtual TEveElementList* getRhoPhiProduct() const;
   virtual TEveElementList* getRhoZProduct() const;
   
   template<class T>
   void build(TEveElementList* oAddTo, T iCaller);
   // ---------- member data --------------------------------
   FWEvePtr<TEveElementList> m_rhoPhiElementsPtr;
   FWEvePtr<TEveElementList> m_rhoZElementsPtr;
   FWEvePtr<TEveElementList> m_compounds;
   const std::type_info* m_itemType;
   long m_objectOffset;

   mutable bool m_rhoPhiNeedsUpdate;
   mutable bool m_rhoZNeedsUpdate;

};


#endif
