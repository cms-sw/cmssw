#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

class FWPRCaloTowerProxyBuilder : public FWDigitSetProxyBuilder {
public:
  FWPRCaloTowerProxyBuilder(void) {}
  ~FWPRCaloTowerProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  FWPRCaloTowerProxyBuilder(const FWPRCaloTowerProxyBuilder&) = delete;                   // stop default
  const FWPRCaloTowerProxyBuilder& operator=(const FWPRCaloTowerProxyBuilder&) = delete;  // stop default

private:
  using FWDigitSetProxyBuilder::build;
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
};

void FWPRCaloTowerProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) {
  const CaloTowerCollection* collection = nullptr;
  iItem->get(collection);
  if (!collection)
    return;

  TEveBoxSet* boxSet = addBoxSetToProduct(product);
  int index = 0;
  for (std::vector<CaloTower>::const_iterator it = collection->begin(); it != collection->end(); ++it) {
    const float* corners = item()->getGeom()->getCorners((*it).id().rawId());
    if (corners == nullptr)
      continue;

    std::vector<float> scaledCorners(24);
    fireworks::energyTower3DCorners(corners, (*it).et(), scaledCorners);

    addBox(boxSet, &scaledCorners[0], iItem->modelInfo(index++).displayProperties());
  }
}

REGISTER_FWPROXYBUILDER(FWPRCaloTowerProxyBuilder, CaloTowerCollection, "PRCaloTower", FWViewType::kISpyBit);
