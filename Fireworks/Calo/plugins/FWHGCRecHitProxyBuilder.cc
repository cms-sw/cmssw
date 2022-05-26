#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "Fireworks/Core/interface/FWGeometry.h"

#include "TEveBoxSet.h"

class FWHGCRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<HGCRecHit> {
public:
  FWHGCRecHitProxyBuilder() {}
  ~FWHGCRecHitProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  FWHGCRecHitProxyBuilder(const FWHGCRecHitProxyBuilder &) = delete;
  const FWHGCRecHitProxyBuilder &operator=(const FWHGCRecHitProxyBuilder &) = delete;

private:
  static constexpr uint8_t gradient_steps = 9;
  static constexpr float gradient[3][gradient_steps] = {{0.2082 * 255,
                                                         0.0592 * 255,
                                                         0.0780 * 255,
                                                         0.0232 * 255,
                                                         0.1802 * 255,
                                                         0.5301 * 255,
                                                         0.8186 * 255,
                                                         0.9956 * 255,
                                                         0.9764 * 255},
                                                        {0.1664 * 255,
                                                         0.3599 * 255,
                                                         0.5041 * 255,
                                                         0.6419 * 255,
                                                         0.7178 * 255,
                                                         0.7492 * 255,
                                                         0.7328 * 255,
                                                         0.7862 * 255,
                                                         0.9832 * 255},
                                                        {0.5293 * 255,
                                                         0.8684 * 255,
                                                         0.8385 * 255,
                                                         0.7914 * 255,
                                                         0.6425 * 255,
                                                         0.4662 * 255,
                                                         0.3499 * 255,
                                                         0.1968 * 255,
                                                         0.0539 * 255}};

  void setItem(const FWEventItem *iItem) override;
  void build(const HGCRecHit &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *) override;
};

void FWHGCRecHitProxyBuilder::setItem(const FWEventItem *iItem) {
  FWProxyBuilderBase::setItem(iItem);
  if (iItem) {
    iItem->getConfig()->keepEntries(true);
    iItem->getConfig()->assertParam("Layer", 0L, 0L, 52L);
    iItem->getConfig()->assertParam("EnergyCutOff", 0.5, 0.2, 5.0);
    iItem->getConfig()->assertParam("Heatmap", true);
    iItem->getConfig()->assertParam("Z+", true);
    iItem->getConfig()->assertParam("Z-", true);
  }
}

void FWHGCRecHitProxyBuilder::build(const HGCRecHit &iData,
                                    unsigned int iIndex,
                                    TEveElement &oItemHolder,
                                    const FWViewContext *) {
  const long layer = item()->getConfig()->value<long>("Layer");
  const double saturation_energy = item()->getConfig()->value<double>("EnergyCutOff");
  const bool heatmap = item()->getConfig()->value<bool>("Heatmap");
  const bool z_plus = item()->getConfig()->value<bool>("Z+");
  const bool z_minus = item()->getConfig()->value<bool>("Z-");

  TEveBoxSet *boxset = new TEveBoxSet();
  if (!heatmap)
    boxset->UseSingleColor();
  boxset->SetPickable(true);
  boxset->SetAntiFlick(true);

  unsigned int ID = iData.id();

  const bool z = (ID >> 25) & 0x1;

  // discard everything thats not at the side that we are intersted in
  if (((z_plus & z_minus) != 1) && (((z_plus | z_minus) == 0) || !(z == z_minus || z == !z_plus)))
    return;

  const float *corners = item()->getGeom()->getCorners(ID);
  const float *parameters = item()->getGeom()->getParameters(ID);
  const float *shapes = item()->getGeom()->getShapePars(ID);

  if (corners == nullptr || parameters == nullptr || shapes == nullptr)
    return;

  const int total_points = parameters[0];
  const bool isScintillator = (total_points == 4);
  const uint8_t type = ((ID >> 28) & 0xF);

  uint8_t ll = layer;
  if (layer > 0) {
    if (layer > 28) {
      if (type == 8) {
        return;
      }
      ll -= 28;
    } else {
      if (type != 8) {
        return;
      }
    }

    if (ll != ((ID >> (isScintillator ? 17 : 20)) & 0x1F))
      return;
  }

  boxset->Reset(isScintillator ? TEveBoxSet::kBT_FreeBox : TEveBoxSet::kBT_Hex, true, 64);

  // Scintillator
  if (isScintillator) {
    const int total_vertices = 3 * total_points;

    std::vector<float> pnts(24);
    for (int i = 0; i < total_points; ++i) {
      pnts[i * 3 + 0] = corners[i * 3];
      pnts[i * 3 + 1] = corners[i * 3 + 1];
      pnts[i * 3 + 2] = corners[i * 3 + 2];

      pnts[(i * 3 + 0) + total_vertices] = corners[i * 3];
      pnts[(i * 3 + 1) + total_vertices] = corners[i * 3 + 1];
      pnts[(i * 3 + 2) + total_vertices] = corners[i * 3 + 2] + shapes[3];
    }
    boxset->AddBox(&pnts[0]);
  }
  // Silicon
  else {
    const int offset = 9;

    float centerX = (corners[6] + corners[6 + offset]) / 2;
    float centerY = (corners[7] + corners[7 + offset]) / 2;
    float radius = fabs(corners[6] - corners[6 + offset]) / 2;
    boxset->AddHex(TEveVector(centerX, centerY, corners[2]), radius, shapes[2], shapes[3]);
  }

  if (heatmap) {
    const uint8_t colorFactor = gradient_steps * (fmin(iData.energy() / saturation_energy, 1.0f));
    boxset->DigitColor(gradient[0][colorFactor], gradient[1][colorFactor], gradient[2][colorFactor]);
  }

  boxset->RefitPlex();

  boxset->CSCTakeAnyParentAsMaster();
  if (!heatmap) {
    boxset->CSCApplyMainColorToMatchingChildren();
    boxset->CSCApplyMainTransparencyToMatchingChildren();
    boxset->SetMainColor(item()->defaultDisplayProperties().color());
    boxset->SetMainTransparency(item()->defaultDisplayProperties().transparency());
  }
  oItemHolder.AddElement(boxset);
}

REGISTER_FWPROXYBUILDER(FWHGCRecHitProxyBuilder, HGCRecHit, "HGC RecHit", FWViewType::kISpyBit);
