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

FWHGCalMultiClusterLegoProxyBuilder::FWHGCalMultiClusterLegoProxyBuilder()
    : FWCaloDataHistProxyBuilder(), m_towers(nullptr) {}

FWHGCalMultiClusterLegoProxyBuilder::~FWHGCalMultiClusterLegoProxyBuilder() {}

void FWHGCalMultiClusterLegoProxyBuilder::build(const FWEventItem *iItem,
                                                TEveElementList *product,
                                                const FWViewContext *vc) {
  m_towers = nullptr;
  if (iItem) {
    iItem->get(m_towers);
    FWCaloDataProxyBuilderBase::build(iItem, product, vc);
  }
}

FWHistSliceSelector *FWHGCalMultiClusterLegoProxyBuilder::instantiateSliceSelector() {
  return new FWHGCalMultiClusterSliceSelector(m_hist, item());
}

void FWHGCalMultiClusterLegoProxyBuilder::fillCaloData() {
  m_hist->Reset();

  if (m_towers) {
    if (item()->defaultDisplayProperties().isVisible()) {
      unsigned int index = 0;
      for (std::vector<reco::HGCalMultiCluster>::const_iterator tower = m_towers->begin(); tower != m_towers->end();
           ++tower, ++index) {
        const FWEventItem::ModelInfo &info = item()->modelInfo(index);
        if (info.displayProperties().isVisible()) {
          addEntryToTEveCaloData(tower->eta(), tower->phi(), tower->pt(), info.isSelected());
        }
      }
    }
  }
}

REGISTER_FWPROXYBUILDER(FWHGCalMultiClusterLegoProxyBuilder,
                        std::vector<reco::HGCalMultiCluster>,
                        "HGCMultiClusterLego",
                        FWViewType::k3DBit | FWViewType::kAllRPZBits | FWViewType::kAllLegoBits);
