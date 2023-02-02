// -*- C++ -*-
//
// Package:     MTD
// Class  :     FWBtlClusterProxyBuilder
//
//
// Original Author:
//         Created:  Thu Nov 24 10:00:00 CET 2022
//

#include "TEvePointSet.h"
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"

class FWBtlClusterProxyBuilder : public FWProxyBuilderBase {
public:
  FWBtlClusterProxyBuilder(void) {}
  ~FWBtlClusterProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  // Disable default copy constructor
  FWBtlClusterProxyBuilder(const FWBtlClusterProxyBuilder&) = delete;
  // Disable default assignment operator
  const FWBtlClusterProxyBuilder& operator=(const FWBtlClusterProxyBuilder&) = delete;

private:
  using FWProxyBuilderBase::build;
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
};

void FWBtlClusterProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) {
  const FTLClusterCollection* clusters = nullptr;

  iItem->get(clusters);

  if (!clusters) {
    fwLog(fwlog::kWarning) << "failed to get the BTL Cluster Collection" << std::endl;
    return;
  }

  const FWGeometry* geom = iItem->getGeom();

  TEvePointSet* pointSet = new TEvePointSet();
  TEveElement* itemHolder = createCompound();
  product->AddElement(itemHolder);

  for (const auto& detSet : *clusters) {
    unsigned int id = detSet.detId();

    if (!geom->contains(id)) {
      fwLog(fwlog::kWarning) << "failed to get BTL geometry element with detid: " << id << std::endl;
      continue;
    }

    const float* pars = geom->getParameters(id);

    for (const auto& cluster : detSet) {
      // --- Get the BTL cluster local position:
      float x_local =
          (cluster.getClusterErrorX() < 0. ? (cluster.x() + 0.5f) * pars[0] + pars[2] : cluster.getClusterPosX());
      float y_local = (cluster.y() + 0.5f) * pars[1] + pars[3];

      const float localPoint[3] = {x_local, y_local, 0.0};

      float globalPoint[3];
      geom->localToGlobal(id, localPoint, globalPoint);

      pointSet->SetNextPoint(globalPoint[0], globalPoint[1], globalPoint[2]);

    }  // cluster loop

  }  // detSet loop

  setupAddElement(pointSet, itemHolder);
}

REGISTER_FWPROXYBUILDER(FWBtlClusterProxyBuilder,
                        FTLClusterCollection,
                        "BTLclusters",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
