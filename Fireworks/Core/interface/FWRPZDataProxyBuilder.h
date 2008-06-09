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
// $Id: FWRPZDataProxyBuilder.h,v 1.6 2008/02/29 21:18:05 chrjones Exp $
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

class FWRPZDataProxyBuilder
{

   public:
      FWRPZDataProxyBuilder();
      virtual ~FWRPZDataProxyBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setItem(const FWEventItem* iItem);
      void build(TEveElementList** product);

      void modelChanges(const FWModelIds&);
   
      void addRhoPhiProj(TEveElement*);
      void addRhoZProj(TEveElement*);

      void clearRhoPhiProjs();
      void clearRhoZProjs();
   
   protected:
      virtual void build(const FWEventItem* iItem, 
			 TEveElementList** product) = 0 ;


      //Override this if you need to special handle selection or other changes
      virtual void modelChanges(const FWModelIds&, TEveElement*);
   
      FWRPZDataProxyBuilder(const FWRPZDataProxyBuilder&); // stop default

      const FWRPZDataProxyBuilder& operator=(const FWRPZDataProxyBuilder&); // stop default

      // ---------- member data --------------------------------
      const FWEventItem* m_item;
      TEveElementList* m_elements;
      std::vector<TEveElement*> m_rhoPhiProjs;
      std::vector<TEveElement*> m_rhoZProjs;
      std::vector<FWModelId> m_ids;
};


#endif
