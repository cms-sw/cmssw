#ifndef Fireworks_Calo_FWCaloTowerProxyBuilder_h
#define Fireworks_Calo_FWCaloTowerProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTowerProxyBuilderBase
//
/**\class FWCaloTowerProxyBuilderBase FWCaloTowerProxyBuilderBase.h Fireworks/Calo/interface/FWCaloTowerProxyBuilderBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 11:28:08 EST 2008
// $Id: FWCaloTowerProxyBuilder.h,v 1.6 2010/05/08 22:03:20 amraktad Exp $
//

#include "Rtypes.h"
#include <string>

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

class TH2F;

class FWCaloTowerProxyBuilderBase : public FWProxyBuilderBase {

public:
   FWCaloTowerProxyBuilderBase();
   virtual ~FWCaloTowerProxyBuilderBase();

   // ---------- const member functions ---------------------
   virtual const std::string histName() const = 0;
   virtual double getEt(const CaloTower&) const = 0;

   virtual bool willHandleInteraction() const { return true; }

   int sliceIndex() const { return m_sliceIndex; }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setSliceIndex(int s) { m_sliceIndex = s; }

private:
   FWCaloTowerProxyBuilderBase(const FWCaloTowerProxyBuilderBase&); // stop default

   const FWCaloTowerProxyBuilderBase& operator=(const FWCaloTowerProxyBuilderBase&); // stop default


   virtual void build(const FWEventItem* iItem,
                      TEveElementList* product, const FWViewContext*);


   virtual void modelChanges(const FWModelIds&, Product*);
   virtual void applyChangesToAllModels(Product*);
   virtual void itemBeingDestroyed(const FWEventItem*);

   // ---------- member data --------------------------------
   TEveCaloDataHist* m_caloData;
   TH2F* m_hist;
   Int_t m_sliceIndex;
   const CaloTowerCollection* m_towers;
};

//
// Ecal
//

class FWECalCaloTowerProxyBuilder : public FWCaloTowerProxyBuilderBase {
public:
   FWECalCaloTowerProxyBuilder() {
      setSliceIndex(0);
   }
   virtual ~FWECalCaloTowerProxyBuilder() {
   }

   // ---------- const member functions ---------------------
   virtual const std::string histName() const {
      return "ECal";
   }

   virtual double getEt(const CaloTower& iTower) const {
      return iTower.emEt();
   }

   REGISTER_PROXYBUILDER_METHODS();
private:
   FWECalCaloTowerProxyBuilder(const FWECalCaloTowerProxyBuilder&); // stop default
   const FWECalCaloTowerProxyBuilder& operator=(const FWECalCaloTowerProxyBuilder&); // stop default
};


//
// Ecal
//

class FWHCalCaloTowerProxyBuilder : public FWCaloTowerProxyBuilderBase {
public:
   FWHCalCaloTowerProxyBuilder() {
      setSliceIndex(1);
   }
   virtual ~FWHCalCaloTowerProxyBuilder(){
   }

   // ---------- const member functions ---------------------
   virtual const std::string histName() const {
      return "HCal";
   }

   virtual double getEt(const CaloTower& iTower) const {
      return iTower.hadEt()+iTower.outerEt();
   }

   REGISTER_PROXYBUILDER_METHODS();
private:
   FWHCalCaloTowerProxyBuilder(const FWHCalCaloTowerProxyBuilder&); // stop default

   const FWHCalCaloTowerProxyBuilder& operator=(const FWHCalCaloTowerProxyBuilder&); // stop default
};


#endif
