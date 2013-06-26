#ifndef Fireworks_Calo_FWCaloDataProxyBuilderBase_h
#define Fireworks_Calo_FWCaloDataProxyBuilderBase_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloDataProxyBuilderBase
// 
/**\class FWCaloDataProxyBuilderBase FWCaloDataProxyBuilderBase.h Fireworks/Calo/interface/FWCaloDataProxyBuilderBase.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon May 31 15:09:19 CEST 2010
// $Id: FWCaloDataProxyBuilderBase.h,v 1.4 2010/10/22 15:34:16 amraktad Exp $
//

// system include files
#include <string>

// user include files

#include "Rtypes.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

// forward declarations

class FWCaloDataProxyBuilderBase : public FWProxyBuilderBase
{
public:
   FWCaloDataProxyBuilderBase();
   virtual ~FWCaloDataProxyBuilderBase();

   // ---------- const member functions ---------------------

   virtual bool willHandleInteraction() const { return true; }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

protected:
   virtual void build(const FWEventItem* iItem,
                      TEveElementList* product, const FWViewContext*);

   virtual void setCaloData(const fireworks::Context&) = 0;
   virtual void fillCaloData() = 0; 
   virtual bool assertCaloDataSlice() = 0;

   // ---------- member data --------------------------------
   TEveCaloData* m_caloData;
   Int_t m_sliceIndex;
   virtual void itemBeingDestroyed(const FWEventItem*);

private:
   FWCaloDataProxyBuilderBase(const FWCaloDataProxyBuilderBase&); // stop default

   const FWCaloDataProxyBuilderBase& operator=(const FWCaloDataProxyBuilderBase&); // stop default

   // ---------- member data --------------------------------


   virtual void modelChanges(const FWModelIds&, Product*);

   void clearCaloDataSelection();
};


#endif
