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
// $Id: FWHFTowerProxyBuilder.cc,v 1.22 2011/09/06 15:07:30 yana Exp $
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
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/fwLog.h"

FWHFTowerProxyBuilderBase::FWHFTowerProxyBuilderBase():
   m_hits(0),
   // m_depth(depth),
   m_vecData(0)
{}

FWHFTowerProxyBuilderBase::~FWHFTowerProxyBuilderBase()
{}

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
      m_caloData->RefSliceInfo(m_sliceIndex).Setup(item()->name().c_str() , 0.,
                                                   item()->defaultDisplayProperties().color(),
                                                   item()->defaultDisplayProperties().transparency());
    
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
   if (iItem)
   {
      iItem->get(m_hits);
      FWCaloDataProxyBuilderBase::build(iItem, el, ctx);
   }
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
               unsigned int rawid = (*it).detid().rawId();
               int tower = fillTowerForDetId(rawid, (*it).energy());
                
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
FWHFTowerProxyBuilderBase::fillTowerForDetId( unsigned int rawid, float val )
{
   using namespace TMath;
   const static float upPhiLimit = Pi() -10*DegToRad() -1e-5;

   TEveCaloData::vCellId_t cellIds;
   const FWGeometry *geom = item()->getGeom();
   if( ! geom->contains( rawid ))
   {
      fwLog( fwlog::kInfo ) << "FWHFTowerProxyBuilderBase cannot get geometry for DetId: "<< rawid << ". Ignored.\n";
      return -1;
   }
     
   const float* corners = geom->getCorners( rawid );
   if( ! corners )
   {
      fwLog( fwlog::kInfo ) << "FWHFTowerProxyBuilderBase cannot get corners for DetId: "<< rawid << ". Ignored.\n";
      return -1;
   }
   
   std::vector<TEveVector> front( 4 );
   float eta[4], phi[4];
   bool plusSignPhi  = false;
   bool minusSignPhi = false;
   int j = 0;
   for( int i = 0; i < 4; ++i )
   {	 
     front[i] = TEveVector( corners[j], corners[j + 1], corners[j + 2] );
     j += 3;
 
     eta[i] = front[i].Eta();
     phi[i] = front[i].Phi();

     // make sure sign around Pi is same as sign of fY
     phi[i] = Sign( phi[i], front[i].fY );

     ( phi[i] >= 0 ) ? plusSignPhi = true :  minusSignPhi = true;
   }

   // check for cell around phi and move up edge to negative side
   if( plusSignPhi && minusSignPhi ) 
   {
      for( int i = 0; i < 4; ++i )
      {
         if( phi[i] >= upPhiLimit ) 
         {
            //  printf("over phi max limit %f \n", phi[i]);
            phi[i] -= TwoPi();
         }
      }
   }
  
   float etaM = -10;
   float etam =  10;
   float phiM = -4;
   float phim =  4;
   for( int i = 0; i < 4; ++i )
   { 
      etam = Min( etam, eta[i] );
      etaM = Max( etaM, eta[i] );
      phim = Min( phim, phi[i] );
      phiM = Max( phiM, phi[i] );
   }

   /*
     if (phiM - phim > 1) 
     printf("!!! [%.2f %.2f] input(%.3f, %.3f, %.3f, %.3f) \n", phim, phiM, phiRef[0] , phiRef[1] , phiRef[2],  phiRef[3]);
   */

   // check if tower is there
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

   // add it if not there 
   if (tower == -1 )
   {
      tower = m_vecData->AddTower(etam, etaM, phim, phiM);
   }


   m_vecData->FillSlice(m_sliceIndex, tower, val);
   return tower; 
}

REGISTER_FWPROXYBUILDER(FWHFTowerProxyBuilderBase, HFRecHitCollection, "HFLego", FWViewType::kLegoHFBit |FWViewType::kAllRPZBits | FWViewType::k3DBit );


