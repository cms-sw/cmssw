#ifndef Fireworks_Core_FW3DLegoEveElementProxyBuilder_h
#define Fireworks_Core_FW3DLegoEveElementProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoEveElementProxyBuilder
//
/**\class FW3DLegoEveElementProxyBuilder FW3DLegoEveElementProxyBuilder.h Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sat Jul  5 11:13:18 EDT 2008
// $Id: FW3DLegoEveElementProxyBuilder.h,v 1.3 2008/11/06 22:05:22 amraktad Exp $
//

// system include files
#include "TEveElement.h"

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

// forward declarations

class FW3DLegoEveElementProxyBuilder : public FW3DLegoDataProxyBuilder
{

public:
   FW3DLegoEveElementProxyBuilder();
   virtual ~FW3DLegoEveElementProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void attach(TEveElement* iElement,
                       TEveCaloDataHist* iHist);
   virtual void build();

private:
   virtual void modelChangesImp(const FWModelIds&);
   virtual void itemChangedImp(const FWEventItem*);
   virtual void applyChangesToAllModels();
   virtual void modelChanges(const FWModelIds&, TEveElement*);
   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product) = 0;
   FW3DLegoEveElementProxyBuilder(const FW3DLegoEveElementProxyBuilder&);    // stop default

   const FW3DLegoEveElementProxyBuilder& operator=(const FW3DLegoEveElementProxyBuilder&);    // stop default

   virtual void itemBeingDestroyedImp(const FWEventItem*);

   // ---------- member data --------------------------------
   TEveElementList m_elementHolder;

};


#endif
