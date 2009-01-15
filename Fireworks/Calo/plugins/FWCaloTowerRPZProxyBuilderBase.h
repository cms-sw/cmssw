// -*- C++ -*-
// $Id: FWCaloTowerProxy3DBuilderBase.h,v 1.1 2009/01/14 19:15:25 amraktad Exp $
//

#ifndef Fireworks_Calo_CaloTowerProxy3DBuilderBase_h
#define Fireworks_Calo_CaloTowerProxy3DBuilderBase_h

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

#endif
