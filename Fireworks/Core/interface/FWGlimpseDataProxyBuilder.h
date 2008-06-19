#ifndef Fireworks_Core_FWGlimpseDataProxyBuilder_h
#define Fireworks_Core_FWGlimpseDataProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseDataProxyBuilder
// 
/**\class FWGlimpseDataProxyBuilder FWGlimpseDataProxyBuilder.h Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sat Jan  5 15:02:03 EST 2008
// $Id: FWGlimpseDataProxyBuilder.h,v 1.9 2008/06/16 18:23:15 dmytro Exp $
//

// system include files
#include <vector>

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilderFactory.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"

// forward declarations
class FWEventItem;
class TEveElementList;
class TEveElement;
class FWModelId;
class TEveCalo3D;

class FWGlimpseDataProxyBuilder
{

   public:
      FWGlimpseDataProxyBuilder();
      virtual ~FWGlimpseDataProxyBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setItem(const FWEventItem* iItem);
      void build(TEveElementList** product);

      void modelChanges(const FWModelIds&);
   
   protected:
      virtual void build(const FWEventItem* iItem, 
			 TEveElementList** product) = 0 ;


      //Override this if you need to special handle selection or other changes
      virtual void modelChanges(const FWModelIds&, TEveElement*);
   
      virtual void itemBeingDestroyed(const FWEventItem*);

      FWGlimpseDataProxyBuilder(const FWGlimpseDataProxyBuilder&); // stop default

      const FWGlimpseDataProxyBuilder& operator=(const FWGlimpseDataProxyBuilder&); // stop default

      // ---------- member data --------------------------------
      const FWEventItem* m_item;
      TEveElementList* m_elements;
      std::vector<FWModelId> m_ids;
};


#endif
