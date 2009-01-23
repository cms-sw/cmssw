#ifndef Fireworks_Core_FWRPZ2DDataProxyBuilder_h
#define Fireworks_Core_FWRPZ2DDataProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZ2DDataProxyBuilder
//
/**\class FWRPZ2DDataProxyBuilder FWRPZ2DDataProxyBuilder.h Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Sat Jan  5 15:02:03 EST 2008
// $Id: FWRPZ2DDataProxyBuilder.h,v 1.12 2008/11/26 16:19:13 chrjones Exp $
//

// system include files
#include <vector>
#include "TEveElement.h"

// user include files
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBaseFactory.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBase.h"

// forward declarations
class FWEventItem;
class TEveElementList;
class TEveElement;
class TEveCalo3D;

class FWRPZ2DDataProxyBuilder : public FWRPZDataProxyBuilderBase
{

public:
   FWRPZ2DDataProxyBuilder();
   virtual ~FWRPZ2DDataProxyBuilder();

   // ---------- const member functions ---------------------
   bool highPriority() const {
      return m_priority;
   }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void buildRhoPhi(TEveElementList** product);
   void buildRhoZ(TEveElementList** product);

   void setHighPriority( bool priority ){
      m_priority = priority;
   }

protected:
   static TEveCalo3D* m_caloRhoPhi;
   static TEveCalo3D* m_caloRhoZ;

private:
   virtual void buildRhoPhi(const FWEventItem* iItem,
                            TEveElementList** product) = 0 ;
   virtual void buildRhoZ(const FWEventItem* iItem,
                          TEveElementList** product) = 0 ;


   //abstract from parent class
   virtual void itemChangedImp(const FWEventItem*) ;
   virtual void itemBeingDestroyedImp(const FWEventItem*);
   virtual void modelChangesImp(const FWModelIds&);
   virtual TEveElementList* getRhoPhiProduct() const;
   virtual TEveElementList* getRhoZProduct() const;


   FWRPZ2DDataProxyBuilder(const FWRPZ2DDataProxyBuilder&);    // stop default

   const FWRPZ2DDataProxyBuilder& operator=(const FWRPZ2DDataProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
   bool m_priority;

   mutable TEveElementList* m_rhoPhiElements;
   mutable TEveElementList* m_rhoZElements;

   mutable bool m_rhoPhiNeedsUpdate;
   mutable bool m_rhoZNeedsUpdate;
};


#endif
