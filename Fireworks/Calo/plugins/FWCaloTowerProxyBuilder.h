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
//

#include "Rtypes.h"
#include <string>

#include "Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h"
#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"


class FWHistSliceSelector;
//
// base
//

class FWCaloTowerProxyBuilderBase : public FWCaloDataHistProxyBuilder
{
public:
   FWCaloTowerProxyBuilderBase();
   virtual ~FWCaloTowerProxyBuilderBase();

   virtual double getEt(const CaloTower&) const = 0;
   
protected:
   virtual void fillCaloData();
   virtual FWHistSliceSelector* instantiateSliceSelector();
   virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);

private:
   FWCaloTowerProxyBuilderBase(const FWCaloTowerProxyBuilderBase&); // stop default
   const FWCaloTowerProxyBuilderBase& operator=(const FWCaloTowerProxyBuilderBase&); // stop default

   const CaloTowerCollection* m_towers;
};

//
// Ecal
//

class FWECalCaloTowerProxyBuilder : public FWCaloTowerProxyBuilderBase {
public:
   FWECalCaloTowerProxyBuilder() {
   }
   virtual ~FWECalCaloTowerProxyBuilder() {
   }

   // ---------- const member functions ---------------------

   virtual double getEt(const CaloTower& iTower) const {
      return iTower.emEt();
   }

   REGISTER_PROXYBUILDER_METHODS();
private:
   FWECalCaloTowerProxyBuilder(const FWECalCaloTowerProxyBuilder&); // stop default
   const FWECalCaloTowerProxyBuilder& operator=(const FWECalCaloTowerProxyBuilder&); // stop default
};


//
// Hcal
//

class FWHCalCaloTowerProxyBuilder : public FWCaloTowerProxyBuilderBase {
public:
   FWHCalCaloTowerProxyBuilder() {
   }
   virtual ~FWHCalCaloTowerProxyBuilder(){
   }

   // ---------- const member functions ---------------------

   virtual double getEt(const CaloTower& iTower) const {
      return iTower.hadEt();
   }

   REGISTER_PROXYBUILDER_METHODS();
private:
   FWHCalCaloTowerProxyBuilder(const FWHCalCaloTowerProxyBuilder&); // stop default

   const FWHCalCaloTowerProxyBuilder& operator=(const FWHCalCaloTowerProxyBuilder&); // stop default
};

//
// HCal Outer
//

class FWHOCaloTowerProxyBuilder : public FWCaloTowerProxyBuilderBase {
public:
   FWHOCaloTowerProxyBuilder() {
   }
   virtual ~FWHOCaloTowerProxyBuilder(){
   }
   
   // ---------- const member functions ---------------------
   
   virtual double getEt(const CaloTower& iTower) const {
      return iTower.outerEt();
   }
   
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWHOCaloTowerProxyBuilder(const FWHOCaloTowerProxyBuilder&); // stop default   
   const FWHOCaloTowerProxyBuilder& operator=(const FWHOCaloTowerProxyBuilder&); // stop default
};

#endif
