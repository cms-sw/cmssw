#ifndef Fireworks_Calo_FWCaloTower3DProxyBuilderBase_h
#define Fireworks_Calo_FWCaloTower3DProxyBuilderBase_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTower3DProxyBuilderBase
// 
/**\class FWCaloTower3DProxyBuilderBase FWCaloTower3DProxyBuilderBase.h Fireworks/Calo/interface/FWCaloTower3DProxyBuilderBase.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 11:28:08 EST 2008
// $Id: FWCaloTower3DProxyBuilderBase.h,v 1.1 2008/12/03 21:05:10 chrjones Exp $
//

// system include files
#include "Rtypes.h"
#include <string>

// user include files
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

// forward declarations
class TH2F;

class FWCaloTower3DProxyBuilderBase : public FW3DDataProxyBuilder {
   
public:
   FWCaloTower3DProxyBuilderBase();
   virtual ~FWCaloTower3DProxyBuilderBase();
   
   // ---------- const member functions ---------------------
   virtual const std::string histName() const = 0;
   virtual double getEt(const CaloTower&) const = 0;
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   virtual void addToScene(TEveElement&, TEveCaloDataHist**);
   
private:
   FWCaloTower3DProxyBuilderBase(const FWCaloTower3DProxyBuilderBase&); // stop default
   
   const FWCaloTower3DProxyBuilderBase& operator=(const FWCaloTower3DProxyBuilderBase&); // stop default
   
   
   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product);
   
   
   virtual void modelChanges(const FWModelIds&, TEveElement*);
   virtual void applyChangesToAllModels(TEveElement* iElements);   
   virtual void itemBeingDestroyed(const FWEventItem*);
   
   // ---------- member data --------------------------------
   TEveCaloDataHist* m_caloData;
   TH2F* m_hist;
   Int_t m_sliceIndex;
   const CaloTowerCollection* m_towers;
};


#endif
