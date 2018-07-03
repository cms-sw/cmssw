#ifndef Fireworks_Core_FWSimpleProxyBuilder_h
#define Fireworks_Core_FWSimpleProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWSimpleProxyBuilder
//
/**\class FWSimpleProxyBuilder FWSimpleProxyBuilder.h Fireworks/Core/interface/FWSimpleProxyBuilder.h

   Description: <one line class summary>

   Usage:s
    <usage>

 */
//
// Original Author:  Chris Jones, AljaMrak-Tadel
//         Created:  Tue March 28  2 09:46:36 EST 2010
//

// system include files
#include <typeinfo>

// user include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

// forward declarations

class FWSimpleProxyBuilder : public FWProxyBuilderBase {

public:
   FWSimpleProxyBuilder(const std::type_info& iType);
   ~FWSimpleProxyBuilder() override;

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------
   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   static std::string typeOfBuilder();

   // ---------- member functions ---------------------------

protected:
   using FWProxyBuilderBase::build;
   void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
   using FWProxyBuilderBase::buildViewType;
   void buildViewType(const FWEventItem* iItem, TEveElementList* product, FWViewType::EType viewType, const FWViewContext*) override;

   //called once for each item in collection, the void* points to the
   // object properly offset in memory
   virtual void build(const void*, unsigned int iIndex, TEveElement& iItemHolder, const FWViewContext*) = 0;
   virtual void buildViewType(const void*, unsigned int iIndex, TEveElement& iItemHolder, FWViewType::EType, const FWViewContext*) = 0;

   void clean() override;
   FWSimpleProxyHelper m_helper;

private:
   FWSimpleProxyBuilder(const FWSimpleProxyBuilder&) = delete; // stop default

   const FWSimpleProxyBuilder& operator=(const FWSimpleProxyBuilder&) = delete; // stop default

   virtual void itemChangedImp(const FWEventItem*);
   
   bool visibilityModelChanges(const FWModelId&, TEveElement*, FWViewType::EType, const FWViewContext*) override;

 
   // ---------- member data --------------------------------
};


#endif
