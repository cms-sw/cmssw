#ifndef Fireworks_Core_FW3DLegoSimpleProxyBuilder_h
#define Fireworks_Core_FW3DLegoSimpleProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoSimpleProxyBuilder
// 
/**\class FW3DLegoSimpleProxyBuilder FW3DLegoSimpleProxyBuilder.h Fireworks/Core/interface/FW3DLegoSimpleProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 16:34:41 EST 2008
// $Id$
//

// system include files
#include <typeinfo>

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

// forward declarations

class FW3DLegoSimpleProxyBuilder : public FW3DLegoEveElementProxyBuilder {
   
public:
   FW3DLegoSimpleProxyBuilder(const std::type_info&);
   //virtual ~FW3DLegoSimpleProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   static std::string typeOfBuilder();

   // ---------- member functions ---------------------------
   
private:
   FW3DLegoSimpleProxyBuilder(const FW3DLegoSimpleProxyBuilder&); // stop default
   
   const FW3DLegoSimpleProxyBuilder& operator=(const FW3DLegoSimpleProxyBuilder&); // stop default
   
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
