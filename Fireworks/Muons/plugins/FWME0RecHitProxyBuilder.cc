#include "TEveGeoNode.h"
#include "TEveGeoShape.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"

class FWME0RecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<ME0RecHit> {
public:
  FWME0RecHitProxyBuilder() {}
  ~FWME0RecHitProxyBuilder() override {}

  bool haveSingleProduct() const override { return false; }

  REGISTER_PROXYBUILDER_METHODS();

  FWME0RecHitProxyBuilder(const FWME0RecHitProxyBuilder&) = delete;
  const FWME0RecHitProxyBuilder& operator=(const FWME0RecHitProxyBuilder&) = delete;

private:
  void buildViewType(const ME0RecHit& iData,
                     unsigned int iIndex,
                     TEveElement& oItemHolder,
                     FWViewType::EType type,
                     const FWViewContext*) override;
};

void FWME0RecHitProxyBuilder::buildViewType(const ME0RecHit& iData,
                                            unsigned int iIndex,
                                            TEveElement& oItemHolder,
                                            FWViewType::EType type,
                                            const FWViewContext*) {
  ME0DetId me0Id = iData.me0Id();
  unsigned int rawid = me0Id.rawId();

  const FWGeometry* geom = item()->getGeom();

  if (!geom->contains(rawid)) {
    fwLog(fwlog::kError) << "failed to get geometry of ME0 roll with detid: " << rawid << std::endl;
    return;
  }

  TEveStraightLineSet* recHitSet = new TEveStraightLineSet;
  recHitSet->SetLineWidth(3);

  if (type == FWViewType::k3D || type == FWViewType::kISpy) {
    TEveGeoShape* shape = geom->getEveShape(rawid);
    shape->SetMainTransparency(75);
    shape->SetMainColor(item()->defaultDisplayProperties().color());
    recHitSet->AddElement(shape);
  }

  float localX = iData.localPosition().x();
  float localY = iData.localPosition().y();
  float localZ = iData.localPosition().z();

  float localXerr = sqrt(iData.localPositionError().xx());
  float localYerr = sqrt(iData.localPositionError().yy());

  float localU1[3] = {localX - localXerr, localY, localZ};

  float localU2[3] = {localX + localXerr, localY, localZ};

  float localV1[3] = {localX, localY - localYerr, localZ};

  float localV2[3] = {localX, localY + localYerr, localZ};

  float globalU1[3];
  float globalU2[3];
  float globalV1[3];
  float globalV2[3];

  FWGeometry::IdToInfoItr det = geom->find(rawid);

  geom->localToGlobal(*det, localU1, globalU1);
  geom->localToGlobal(*det, localU2, globalU2);
  geom->localToGlobal(*det, localV1, globalV1);
  geom->localToGlobal(*det, localV2, globalV2);

  recHitSet->AddLine(globalU1[0], globalU1[1], globalU1[2], globalU2[0], globalU2[1], globalU2[2]);

  recHitSet->AddLine(globalV1[0], globalV1[1], globalV1[2], globalV2[0], globalV2[1], globalV2[2]);

  setupAddElement(recHitSet, &oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWME0RecHitProxyBuilder,
                        ME0RecHit,
                        "ME0 RecHits",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
