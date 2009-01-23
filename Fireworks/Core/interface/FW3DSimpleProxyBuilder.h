#ifndef Fireworks_Core_FW3DSimpleProxyBuilder_h
#define Fireworks_Core_FW3DSimpleProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DSimpleProxyBuilder
//
/**\class FW3DSimpleProxyBuilder FW3DSimpleProxyBuilder.h Fireworks/Core/interface/FW3DSimpleProxyBuilder.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 09:46:36 EST 2008
// $Id: FW3DSimpleProxyBuilder.h,v 1.1 2008/12/04 15:24:37 dmytro Exp $
//

// system include files
#include <typeinfo>

// user include files
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

// forward declarations

class FW3DSimpleProxyBuilder : public FW3DDataProxyBuilder {

public:
   FW3DSimpleProxyBuilder(const std::type_info& iType);
   virtual ~FW3DSimpleProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------
   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   static std::string typeOfBuilder();

   // ---------- member functions ---------------------------

private:
   FW3DSimpleProxyBuilder(const FW3DSimpleProxyBuilder&); // stop default

   const FW3DSimpleProxyBuilder& operator=(const FW3DSimpleProxyBuilder&); // stop default

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
