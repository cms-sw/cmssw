// -*- C++ -*-
// $Id: FWCalCaloTowerProxy3DBuilderBase.h,v 1.1 2009/01/14 12:06:45 amraktad Exp $
//

#ifndef Fireworks_Calo_CaloTowerProxy3DBuilderBase_h
#define Fireworks_Calo_CaloTowerProxy3DBuilderBase_h

class TH2F;
class TEveCaloDataHist;

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

class FWCaloTowerProxy3DBuilderBase : public FWRPZDataProxyBuilder
{
public:
   FWCaloTowerProxy3DBuilderBase(): m_handleEcal(true), m_histName("blank"), m_hist(0), m_sliceIndex(-1), m_towers(0) { setHighPriority( true ); }
   FWCaloTowerProxy3DBuilderBase(bool handleEcal, const char* name): m_handleEcal(handleEcal), m_histName(name), m_hist(0), m_sliceIndex(-1), m_towers(0) { setHighPriority( true ); }
   virtual ~FWCaloTowerProxy3DBuilderBase() {}

protected:
   void itemBeingDestroyedImp(const FWEventItem*);

   const CaloTowerCollection* m_towers;

private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   virtual void modelChanges(const FWModelIds& iIds,
                             TEveElement* iElements );
   virtual void applyChangesToAllModels(TEveElement* iElements);

   FWCaloTowerProxy3DBuilderBase(const FWCaloTowerProxy3DBuilderBase&); // stop default
   const FWCaloTowerProxy3DBuilderBase& operator=(const FWCaloTowerProxy3DBuilderBase&); // stop default

   // ---------- member data --------------------------------
   static TEveCaloDataHist* m_data;

   bool         m_handleEcal;
   const char*  m_histName;
   TH2F*        m_hist;
   Int_t        m_sliceIndex;
};

#endif
