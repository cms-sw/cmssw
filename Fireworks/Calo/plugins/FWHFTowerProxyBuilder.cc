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
// $Id: FWHFTowerProxyBuilder.cc,v 1.4 2010/06/02 19:08:33 amraktad Exp $
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


FWHFTowerProxyBuilder::FWHFTowerProxyBuilder():
   m_hits(0),
   m_depth(-1)
{
}


FWHFTowerProxyBuilder::~FWHFTowerProxyBuilder()
{
}

//
// member functions
//
void
FWHFTowerProxyBuilder::setCaloData(const fireworks::Context&)
{
  m_caloData = context().getCaloDataHF();
}

void
FWHFTowerProxyBuilder::addSliceSelector()
{
   FWFromTEveCaloDataSelector* sel = reinterpret_cast<FWFromTEveCaloDataSelector*>(m_caloData->GetUserData());
   sel->addSliceSelector(m_sliceIndex, new FWHFTowerSliceSelector(m_hist,item()));
}

void
FWHFTowerProxyBuilder::build(const FWEventItem* iItem,
                                  TEveElementList* el, const FWViewContext* ctx)
{
   m_hits=0;
   if (iItem) iItem->get(m_hits);
   FWCaloDataHistProxyBuilder::build(iItem, el, ctx);
}

void
FWHFTowerProxyBuilder::fillCaloData()
{
   m_hist->Reset();
   if (m_hits)
   {
      TEveCaloData::vCellId_t& selected = m_caloData->GetCellsSelected();

      if(item()->defaultDisplayProperties().isVisible()) {
         assert(item()->size() >= m_hits->size());
         unsigned int index=0;
         for(HFRecHitCollection::const_iterator it = m_hits->begin(); it != m_hits->end(); ++it,++index) {
            const FWEventItem::ModelInfo& info = item()->modelInfo(index);
            if(info.displayProperties().isVisible()) {
               Float_t energy = (*it).energy();
               std::vector<TEveVector> corners = item()->getGeom()->getPoints((*it).detid().rawId());
               if( corners.empty() ) {
                  break;
               }
               HcalDetId id ((*it).detid().rawId()); 
	       TEveVector centre = corners[0] + corners[1] + corners[2] + corners[3] + corners[4] + corners[5] + corners[6] + corners[7];
	       centre *= 1.0f / 8.0f;
               if(id.depth() == m_depth)
               {
                  int bin = (m_hist)->Fill(centre.Eta(),centre.Phi(), energy*0.2);  
                  int sbin = m_hist->FindBin(centre.Eta(),centre.Phi());
                  if (bin != sbin) fwLog(fwlog::kWarning) << "FWHFShortTowerProxyBuilder::fillHistogram() "<< histName() <<" for depth ["<< m_depth <<"]: discrepancy between filled and search bin "<< bin << " != " << sbin << std::endl;

                  if(info.isSelected())
                  {
                     fwLog(fwlog::kDebug) << " [" << m_depth <<"]  fiber selected at (" <<centre.Eta() << ", " << centre.Phi() << "value "<< energy << "\n";
                     selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(centre.Eta(),centre.Phi()),m_sliceIndex));
                  } 
               }
            }
         }
      }
   }
}


REGISTER_FWPROXYBUILDER(FWHFShortTowerProxyBuilder, HFRecHitCollection, "HFShort", FWViewType::kLegoHFBit);
REGISTER_FWPROXYBUILDER(FWHFLongTowerProxyBuilder , HFRecHitCollection, "HFLong" , FWViewType::kLegoHFBit);


