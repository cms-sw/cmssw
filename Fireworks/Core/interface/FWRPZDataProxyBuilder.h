#ifndef Fireworks_Core_FWRPZDataProxyBuilder_h
#define Fireworks_Core_FWRPZDataProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZDataProxyBuilder
// 
/**\class FWRPZDataProxyBuilder FWRPZDataProxyBuilder.h Fireworks/Core/interface/FWRPZDataProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sat Jan  5 15:02:03 EST 2008
// $Id: FWRPZDataProxyBuilder.h,v 1.9 2008/06/16 18:23:15 dmytro Exp $
//

// system include files
#include <vector>

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderFactory.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"

// forward declarations
class FWEventItem;
class TEveElementList;
class TEveElement;
class FWModelId;
class TEveCalo3D;

class FWRPZDataProxyBuilder
{

   public:
      FWRPZDataProxyBuilder();
      virtual ~FWRPZDataProxyBuilder();

      // ---------- const member functions ---------------------
      bool highPriority() const { return m_priority; }
	
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setItem(const FWEventItem* iItem);
      void build(TEveElementList** product);

      void modelChanges(const FWModelIds&);
   
      void addRhoPhiProj(TEveElement*);
      void addRhoZProj(TEveElement*);

      void clearRhoPhiProjs();
      void clearRhoZProjs();
      void setHighPriority( bool priority ){ m_priority = priority; }
   
   protected:
      virtual void build(const FWEventItem* iItem, 
			 TEveElementList** product) = 0 ;


      //Override this if you need to special handle selection or other changes
      virtual void modelChanges(const FWModelIds&, TEveElement*);
   
      virtual void itemBeingDestroyed(const FWEventItem*);

      FWRPZDataProxyBuilder(const FWRPZDataProxyBuilder&); // stop default

      const FWRPZDataProxyBuilder& operator=(const FWRPZDataProxyBuilder&); // stop default

      // ---------- member data --------------------------------
      bool m_priority;
      const FWEventItem* m_item;
      TEveElementList* m_elements;
      std::vector<TEveElement*> m_rhoPhiProjs;
      std::vector<TEveElement*> m_rhoZProjs;
      std::vector<FWModelId> m_ids;
   
      static TEveCalo3D* m_calo3d;
};


#endif
