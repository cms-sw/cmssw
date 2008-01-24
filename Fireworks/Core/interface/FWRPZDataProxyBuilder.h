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
// $Id: FWRPZDataProxyBuilder.h,v 1.3 2008/01/22 21:08:57 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWModelChangeSignal.h"

// forward declarations
class FWEventItem;
class TEveElementList;
class TEveElement;

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
   
      void setRhoPhiProj(TEveElement*);
      void setRhoZProj(TEveElement*);

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
      TEveElement* m_rhoPhiProj;
      TEveElement* m_rhoZProj;
};


#endif
