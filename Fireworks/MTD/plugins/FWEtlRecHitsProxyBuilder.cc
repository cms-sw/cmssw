
// -*- C++ -*-
//
// Package:     MTD
// Class  :     FWEtlRecHitProxyBuilder
//
//
// Original Author:
//         Created:  Tue Dec 6 17:00:00 CET 2022
//

#include "TEvePointSet.h"
#include "TEveCompound.h"

#include "TEveBox.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

class FWEtlRecHitProxyBuilder : public FWProxyBuilderBase {
public:
  FWEtlRecHitProxyBuilder(void) {}
  ~FWEtlRecHitProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  // Disable default copy constructor
  FWEtlRecHitProxyBuilder(const FWEtlRecHitProxyBuilder&) = delete;
  // Disable default assignment operator
  const FWEtlRecHitProxyBuilder& operator=(const FWEtlRecHitProxyBuilder&) = delete;

private:
  using FWProxyBuilderBase::build;
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
};

void FWEtlRecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) {

  const FTLRecHitCollection* recHits = nullptr;

  iItem->get(recHits);

  if (!recHits) {
    fwLog(fwlog::kWarning) << "failed to get the FTLRecHitCollection" << std::endl;
    return;
  }

  for (const auto& hit : *recHits) {

    unsigned int id = hit.id().rawId();

    const FWGeometry* geom = iItem->getGeom();
    const float* pars = geom->getParameters(id);

    TEveElement* itemHolder = createCompound();
    product->AddElement(itemHolder);

    TEvePointSet* pointSet = new TEvePointSet;

    if (!geom->contains(id)) {
      fwLog(fwlog::kWarning) << "failed to get geometry of FTLRecHit with detid: " << id << std::endl;
      continue;
    }

    // --- Get the ETL RecHit local position:
    float x_local = (hit.row() + 0.5f) * pars[0] + pars[2];
    float y_local = (hit.column() + 0.5f) * pars[1] + pars[3]; 

    const float localPoint[3] = {x_local, y_local, 0.0};

    float globalPoint[3];
    geom->localToGlobal(id, localPoint, globalPoint);

    pointSet->SetNextPoint(globalPoint[0], globalPoint[1], globalPoint[2]);

    setupAddElement(pointSet, itemHolder);

  }  // recHits loop

}

REGISTER_FWPROXYBUILDER(FWEtlRecHitProxyBuilder,
                        FTLRecHitCollection,
                        "ETLrechits",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
