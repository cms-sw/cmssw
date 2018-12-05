// system includes
#include <cmath>

// user includes
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "TH2F.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"

#include "Fireworks/Core/interface/fw3dlego_xbins.h"
#include "Fireworks/Calo/plugins/FWHGCalMultiClusterLegoProxyBuilder.h"
#include "Fireworks/Calo/plugins/FWHGCalMultiClusterSliceSelector.h"

#include "FWCore/Common/interface/EventBase.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "TEveCompound.h"

FWHGCalMultiClusterLegoProxyBuilder::FWHGCalMultiClusterLegoProxyBuilder() : FWCaloDataHistProxyBuilder(), m_helper(typeid(reco::HGCalMultiCluster))
{
}

FWHGCalMultiClusterLegoProxyBuilder::~FWHGCalMultiClusterLegoProxyBuilder()
{
}

void FWHGCalMultiClusterLegoProxyBuilder::build(const FWEventItem *iItem, TEveElementList *product, const FWViewContext *vc)
{
   setCaloData(iItem->context());
   assertCaloDataSlice();

   //--------------------------------------------------

   m_hist->Reset();
   if (item()->defaultDisplayProperties().isVisible())
   {
      size_t size = iItem->size();
      TEveElement::List_i pIdx = product->BeginChildren();
      for (int index = 0; index < static_cast<int>(size); ++index)
      {
         TEveElement *itemHolder = nullptr;
         if (index < product->NumChildren())
         {
            itemHolder = *pIdx;
            itemHolder->SetRnrSelfChildren(true, true);
            ++pIdx;
         }
         else
         {
            itemHolder = createCompound();
            product->AddElement(itemHolder);
         }
         if (iItem->modelInfo(index).displayProperties().isVisible())
         {
            const void *modelData = iItem->modelData(index);
            build(*reinterpret_cast<const reco::HGCalMultiCluster *>(m_helper.offsetObject(modelData)), index, *itemHolder, vc);
         }
      }
   }

   m_caloData->SetSliceColor(m_sliceIndex, item()->defaultDisplayProperties().color());
   m_caloData->SetSliceTransparency(m_sliceIndex, item()->defaultDisplayProperties().transparency());
   m_caloData->DataChanged();
   m_caloData->CellSelectionChanged();
}

void FWHGCalMultiClusterLegoProxyBuilder::build(const reco::HGCalMultiCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc)
{
   const FWEventItem::ModelInfo &info = item()->modelInfo(iIndex);
   if (info.displayProperties().isVisible())
      addEntryToTEveCaloData(iData.eta(), iData.phi(), iData.pt(), info.isSelected());
}

FWHistSliceSelector *
FWHGCalMultiClusterLegoProxyBuilder::instantiateSliceSelector()
{
   return new FWHGCalMultiClusterSliceSelector(m_hist, item());
}

// std::string
// FWHGCalMultiClusterLegoProxyBuilder::typeOfBuilder()
// {
//    return std::string("simple#");
// }

REGISTER_FWPROXYBUILDER(FWHGCalMultiClusterLegoProxyBuilder, std::vector<reco::HGCalMultiCluster>, "CaloFTW2", FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits);