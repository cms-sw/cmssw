// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWHFTowerProxyBuilder
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Mon May 31 16:41:27 CEST 2010
// $Id: FWHFTowerProxyBuilder.cc,v 1.9 2010/06/08 18:43:15 amraktad Exp $
//

// system include files

// user include files
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "TH2F.h"

#include "Fireworks/Calo/plugins/FWHFTowerProxyBuilder.h"
#include "Fireworks/Calo/plugins/FWHFTowerSliceSelector.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/fwLog.h"

  

FWHFTowerProxyBuilderBase::FWHFTowerProxyBuilderBase():
   m_hits(0),
   // m_depth(depth),
   m_vecData(0)
{
}


FWHFTowerProxyBuilderBase::~FWHFTowerProxyBuilderBase()
{
}

//
// member functions

void
FWHFTowerProxyBuilderBase::setCaloData(const fireworks::Context& ctx)
{
   m_vecData  = ctx.getCaloDataHF();// cached to avoid casting
   m_caloData = m_vecData;
}

bool
FWHFTowerProxyBuilderBase::assertCaloDataSlice()
{
  if (m_sliceIndex == -1)
  {
    m_sliceIndex = m_vecData->AddSlice();
    // printf("add slice %d \n",m_sliceIndex  );
    m_caloData->RefSliceInfo(m_sliceIndex).Setup(sliceName().c_str(), 0., item()->defaultDisplayProperties().color());
    
    
    // add new selector
    FWFromTEveCaloDataSelector* sel = 0;
    if (m_caloData->GetUserData())
    {
      FWFromEveSelectorBase* base = reinterpret_cast<FWFromEveSelectorBase*>(m_caloData->GetUserData());
      assert(0!=base);
      sel = dynamic_cast<FWFromTEveCaloDataSelector*> (base);
      assert(0!=sel);
    }
    else
    {
      sel = new FWFromTEveCaloDataSelector(m_caloData);
      //make sure it is accessible via the base class
      m_caloData->SetUserData(static_cast<FWFromEveSelectorBase*>(sel));
    }
    
    sel->addSliceSelector(m_sliceIndex, new FWHFTowerSliceSelector(item(), m_vecData));
    
    return true;
  }
  return false;
}



void
FWHFTowerProxyBuilderBase::build(const FWEventItem* iItem,
                                  TEveElementList* el, const FWViewContext* ctx)
{
   m_hits=0;
   if (iItem) iItem->get(m_hits);
   FWCaloDataProxyBuilderBase::build(iItem, el, ctx);
}

void
FWHFTowerProxyBuilderBase::itemBeingDestroyed(const FWEventItem* iItem)
{
  
   if(0!=m_hits) {

      //reset values for this slice
      std::vector<float>& sliceVals = m_vecData->GetSliceVals(m_sliceIndex);
      for (std::vector<float>::iterator i = sliceVals.begin(); i!= sliceVals.end(); ++i)
      {
         *i = 0;
      }


   }
   FWCaloDataProxyBuilderBase::itemBeingDestroyed(iItem);
}

void
FWHFTowerProxyBuilderBase::fillCaloData()
{
   //reset values for this slice
   std::vector<float>& sliceVals = m_vecData->GetSliceVals(m_sliceIndex);
   for (std::vector<float>::iterator i = sliceVals.begin(); i!= sliceVals.end(); ++i)
   {
      *i = 0;
   }

   if (m_hits)
   {
      TEveCaloData::vCellId_t& selected = m_vecData->GetCellsSelected();

      if(item()->defaultDisplayProperties().isVisible()) {
         assert(item()->size() >= m_hits->size());

         unsigned int index=0;
         TEveCaloData::vCellId_t cellId;
         for(HFRecHitCollection::const_iterator it = m_hits->begin(); it != m_hits->end(); ++it,++index) 
         {
            const FWEventItem::ModelInfo& info = item()->modelInfo(index);
            if(info.displayProperties().isVisible())
            {
               HcalDetId detId = (*it).detid().rawId();
               int tower = fillTowerForDetId(detId, (*it).energy());
                
               if(info.isSelected())
               {
                  selected.push_back(TEveCaloData::CellId_t(tower, m_sliceIndex));
               } 
            }
         }
      }
   }
}


int
FWHFTowerProxyBuilderBase::fillTowerForDetId(HcalDetId& detId, float val)
{
   const static float PI = TMath::Pi();

   TEveCaloData::vCellId_t cellIds;
   std::vector<TEveVector> pnts = item()->getGeom()->getPoints(detId.rawId());
 
   // complication with float to duouble comaprison of TMath::Pi()
   float eta[4], phi[4];
   bool plusSignPhi  = false;
   bool minusSignPhi = false;

   for (int i = 0; i < 4; ++i)
   {
      eta[i] =  pnts[i].Eta();
      phi[i] =  pnts[i].Phi();
      if (phi[i] >= 0)
         plusSignPhi = true;
      else
         minusSignPhi  = true;
   }
 
   // check if phi sign is changin
   // take sides depending on average
   if (plusSignPhi && minusSignPhi)
   {
      if (TMath::Abs(phi[0]) > 1)
      {
          fwLog(fwlog::kWarning) << "FWHFTowerProxyBuilderBase::fillData() cell changing sign at PI edges " << phi[0] << ", " << phi[1] << ", " << phi[2] << ", " << phi[3] << std::endl;
      }

      float phiMean = 0;
      int n = 0;
      for (int i = 0; i < 4; ++i)
      {
         if (TMath::Abs(TMath::Abs(phi[i]) - PI) > 1e-3)
         {
            phiMean += phi[i];
            ++n;
         }
      }
      if (phiMean >= 0)
         minusSignPhi = false;
      else
         plusSignPhi = false;
   }

   float etaM = -10;
   float etam =  10;
   float phiM = -4;
   float phim =  4;
   float origPhi =10;
   for (int i = 0; i < 4; ++i)
   {
      if (phi[i] < -1 && plusSignPhi)
      {
         origPhi =  phi[i];
         phi[i] += TMath::TwoPi();
         phi[i] = TMath::Min(phi[i], PI);
         fwLog(fwlog::kWarning) << "FWHFTowerProxyBuilderBase::fillData() positive average, fix negative phi sign " <<  origPhi  << " -> " << phi[i] << std::endl;
      }
      else if (phi[i] > 1 && minusSignPhi)
      {
         origPhi =  phi[i];
         phi[i] -=  TMath::TwoPi();
         phi[i]  = TMath::Max(phi[i], -PI);
         fwLog(fwlog::kWarning) << "FWHFTowerProxyBuilderBase::fillData() negative average fix positive phi sign n " <<  origPhi  << " -> " << phi[i] << std::endl;
      }

      etam = TMath::Min(etam, eta[i]);
      etaM = TMath::Max(etaM, eta[i]);
      phim = TMath::Min(phim, phi[i]);
      phiM = TMath::Max(phiM, phi[i]);
   }

   Float_t ceta = (etam+etaM)*0.5;
   Float_t cphi = (phim+phiM)*0.5;

   int tower = -1;
  
   int idx = 0;
   for ( TEveCaloData::vCellGeom_i i = m_vecData->GetCellGeom().begin(); i!= m_vecData->GetCellGeom().end(); ++i, ++idx)
   {
      const TEveCaloData::CellGeom_t &cg = *i;
      if ((ceta > cg.fEtaMin && ceta < cg.fEtaMax) && (cphi > cg.fPhiMin && cphi < cg.fPhiMax))
      {
         tower = idx;
         break;
      }
   }

   if (tower == -1 )
   {
      tower = m_vecData->AddTower(etam, etaM, phim, phiM);
   }

   m_vecData->FillSlice(m_sliceIndex, tower, val);

   return tower; 
}


REGISTER_FWPROXYBUILDER(FWHFTowerProxyBuilderBase, HFRecHitCollection, "HFLego", FWViewType::kLegoHFBit);


