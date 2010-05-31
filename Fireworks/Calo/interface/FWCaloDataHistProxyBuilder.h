#ifndef Fireworks_Calo_FWCaloDataHistProxyBuilder_h
#define Fireworks_Calo_FWCaloDataHistProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloDataHistProxyBuilder
// 
/**\class FWCaloDataHistProxyBuilder FWCaloDataHistProxyBuilder.h Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon May 31 15:09:19 CEST 2010
// $Id$
//

// system include files
#include <string>

// user include files

#include "Rtypes.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

// forward declarations

class FWCaloDataHistProxyBuilder : public FWProxyBuilderBase
{
public:
   FWCaloDataHistProxyBuilder();
   virtual ~FWCaloDataHistProxyBuilder();

   // ---------- const member functions ---------------------

   virtual const std::string histName() const { return "sliceHist"; }

   virtual bool willHandleInteraction() const { return true; }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

protected:
   virtual void build(const FWEventItem* iItem,
                      TEveElementList* product, const FWViewContext*);

   virtual void setCaloData(const fireworks::Context&) = 0;
   virtual void fillCaloData() = 0; 

   // ---------- member data --------------------------------
   TEveCaloDataHist* m_caloData;
   TH2F* m_hist;
   Int_t m_sliceIndex;

private:
   FWCaloDataHistProxyBuilder(const FWCaloDataHistProxyBuilder&); // stop default

   const FWCaloDataHistProxyBuilder& operator=(const FWCaloDataHistProxyBuilder&); // stop default

   // ---------- member data --------------------------------


   virtual void modelChanges(const FWModelIds&, Product*);
   virtual void applyChangesToAllModels(Product*);
   virtual void itemBeingDestroyed(const FWEventItem*);

   void clearCaloDataSelection();
   bool assertHistogram();

};


#endif
