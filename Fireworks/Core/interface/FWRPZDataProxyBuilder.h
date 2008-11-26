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
// $Id: FWRPZDataProxyBuilder.h,v 1.14 2008/11/06 22:05:23 amraktad Exp $
//

// system include files
#include <vector>
#include "TEveElement.h"

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBaseFactory.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBase.h"

// forward declarations
class FWEventItem;
class TEveElementList;
class TEveElement;
class FWModelId;
class TEveCalo3D;

class FWRPZDataProxyBuilder : public FWRPZDataProxyBuilderBase
{

   public:
      FWRPZDataProxyBuilder();
      virtual ~FWRPZDataProxyBuilder();

      // ---------- const member functions ---------------------
      bool highPriority() const { return m_priority; }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void build();

      void setHighPriority( bool priority ){ m_priority = priority; }

   protected:
      static TEveCalo3D* m_calo3d;

   virtual void itemChangedImp(const FWEventItem*) ;
   virtual void itemBeingDestroyedImp(const FWEventItem*);
   virtual void modelChangesImp(const FWModelIds&);
private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product) = 0 ;


      //abstract functions from the base class
      virtual TEveElementList* getRhoPhiProduct() const;
      virtual TEveElementList* getRhoZProduct() const;

      FWRPZDataProxyBuilder(const FWRPZDataProxyBuilder&); // stop default

      const FWRPZDataProxyBuilder& operator=(const FWRPZDataProxyBuilder&); // stop default

      // ---------- member data --------------------------------
      bool m_priority;
      mutable TEveElementList* m_elements;

      mutable bool m_needsUpdate;
};


#endif
