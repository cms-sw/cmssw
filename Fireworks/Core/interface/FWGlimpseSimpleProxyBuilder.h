#ifndef Fireworks_Core_FWGlimpseSimpleProxyBuilder_h
#define Fireworks_Core_FWGlimpseSimpleProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseSimpleProxyBuilder
// 
/**\class FWGlimpseSimpleProxyBuilder FWGlimpseSimpleProxyBuilder.h Fireworks/Core/interface/FWGlimpseSimpleProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 09:46:36 EST 2008
// $Id$
//

// system include files
#include <typeinfo>

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

// forward declarations

class FWGlimpseSimpleProxyBuilder : public FWGlimpseDataProxyBuilder {
   
public:
   FWGlimpseSimpleProxyBuilder(const std::type_info& iType);
   virtual ~FWGlimpseSimpleProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   static std::string typeOfBuilder();
   
   // ---------- member functions ---------------------------
   
private:
   FWGlimpseSimpleProxyBuilder(const FWGlimpseSimpleProxyBuilder&); // stop default
   
   const FWGlimpseSimpleProxyBuilder& operator=(const FWGlimpseSimpleProxyBuilder&); // stop default
   
   virtual void itemChangedImp(const FWEventItem*);
   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product);
   
   //called once for each item in collection, the void* points to the 
   // object properly offset in memory
   virtual void build(const void*, unsigned int iIndex, TEveElement& iItemHolder) const = 0;
   
   // ---------- member data --------------------------------
   FWSimpleProxyHelper m_helper;

};


#endif
