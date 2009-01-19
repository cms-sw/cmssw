// -*- C++ -*-
// $Id: FWCaloTowerRPZProxyBuilderBase.h,v 1.1 2009/01/15 16:28:01 amraktad Exp $
//

#ifndef Fireworks_Calo_CaloTowerProxyRPZBuilder_h
#define Fireworks_Calo_CaloTowerProxyRPZBuilder_h

class TH2F;
class TEveCaloDataHist;

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

class FWCaloTowerRPZProxyBuilderBase : public FWRPZDataProxyBuilder
{
public:
   FWCaloTowerRPZProxyBuilderBase(): m_towers(0), m_handleEcal(true), m_histName("blank"), m_hist(0), m_sliceIndex(-1) { setHighPriority( true ); }
   FWCaloTowerRPZProxyBuilderBase(bool handleEcal, const char* name): m_towers(0), m_handleEcal(handleEcal), m_histName(name), m_hist(0), m_sliceIndex(-1) { setHighPriority( true ); }
   virtual ~FWCaloTowerRPZProxyBuilderBase() {}

protected:
   void itemBeingDestroyedImp(const FWEventItem*);

   const CaloTowerCollection* m_towers;

private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   virtual void modelChanges(const FWModelIds& iIds,
                             TEveElement* iElements );
   virtual void applyChangesToAllModels(TEveElement* iElements);

   FWCaloTowerRPZProxyBuilderBase(const FWCaloTowerRPZProxyBuilderBase&); // stop default
   const FWCaloTowerRPZProxyBuilderBase& operator=(const FWCaloTowerRPZProxyBuilderBase&); // stop default

   // ---------- member data --------------------------------
   static TEveCaloDataHist* m_data;

   bool         m_handleEcal;
   const char*  m_histName;
   TH2F*        m_hist;
   Int_t        m_sliceIndex;
};

//
// ECal
// 

class FWECalCaloTowerRPZProxyBuilder : public FWCaloTowerRPZProxyBuilderBase
{
public:
   FWECalCaloTowerRPZProxyBuilder(): FWCaloTowerRPZProxyBuilderBase(true, "ecal3D") {}
   virtual ~FWECalCaloTowerRPZProxyBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWECalCaloTowerRPZProxyBuilder(const FWECalCaloTowerRPZProxyBuilder&); // stop default

   const FWECalCaloTowerRPZProxyBuilder& operator=(const FWECalCaloTowerRPZProxyBuilder&); // stop default
};

//
// HCal
//

class FWHCalCaloTowerRPZProxyBuilder : public FWCaloTowerRPZProxyBuilderBase
{
public:
   FWHCalCaloTowerRPZProxyBuilder(): FWCaloTowerRPZProxyBuilderBase(false, "hcal3D") {}
   virtual ~FWHCalCaloTowerRPZProxyBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWHCalCaloTowerRPZProxyBuilder(const FWHCalCaloTowerRPZProxyBuilder&); // stop default

   const FWHCalCaloTowerRPZProxyBuilder& operator=(const FWHCalCaloTowerRPZProxyBuilder&); // stop default
};



#endif
