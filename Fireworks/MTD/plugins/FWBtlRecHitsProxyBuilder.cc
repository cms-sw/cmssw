// -*- C++ -*-
//
// Package:     MTD
// Class  :     FWBtlRecHitProxyBuilder
//
//
// Original Author:
//         Created:  Tue Dec 6 17:00:00 CET 2022
//

#include "TEvePointSet.h"
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

class FWBtlRecHitProxyBuilder : public FWProxyBuilderBase {
public:
  FWBtlRecHitProxyBuilder(void) {}
  ~FWBtlRecHitProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  // Disable default copy constructor
  FWBtlRecHitProxyBuilder(const FWBtlRecHitProxyBuilder&) = delete;
  // Disable default assignment operator
  const FWBtlRecHitProxyBuilder& operator=(const FWBtlRecHitProxyBuilder&) = delete;

private:
  using FWProxyBuilderBase::build;
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
};

void FWBtlRecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) {
  const FTLRecHitCollection* recHits = nullptr;

  iItem->get(recHits);

  if (!recHits) {
    fwLog(fwlog::kWarning) << "failed to get the BTL RecHit Collection" << std::endl;
    return;
  }

  const FWGeometry* geom = iItem->getGeom();

  for (const auto& hit : *recHits) {
    unsigned int id = hit.id().rawId();

    if (!geom->contains(id)) {
      fwLog(fwlog::kWarning) << "failed to get BTL geometry element with detid: " << id << std::endl;
      continue;
    }

    TEvePointSet* pointSet = new TEvePointSet();

    TEveElement* itemHolder = createCompound();
    product->AddElement(itemHolder);

    const float* pars = geom->getParameters(id);

    // --- Get the BTL RecHit local position:
    float x_local = (hit.positionError() > 0. ? hit.position() : (hit.row() + 0.5f) * pars[0] + pars[2]);
    float y_local = (hit.column() + 0.5f) * pars[1] + pars[3];

    const float localPoint[3] = {x_local, y_local, 0.0};

    float globalPoint[3];
    geom->localToGlobal(id, localPoint, globalPoint);

    pointSet->SetNextPoint(globalPoint[0], globalPoint[1], globalPoint[2]);

    setupAddElement(pointSet, itemHolder);

  }  // recHits loop
}

REGISTER_FWPROXYBUILDER(FWBtlRecHitProxyBuilder,
                        FTLRecHitCollection,
                        "BTLrechits",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
