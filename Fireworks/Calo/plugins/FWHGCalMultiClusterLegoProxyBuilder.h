#ifndef Fireworks_Calo_FWCaloParticleTowerProxyBuilder_h
#define Fireworks_Calo_FWCaloParticleTowerProxyBuilder_h

#include "Rtypes.h"
#include <string>
#include <typeinfo>

#include "Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h"
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

class FWHistSliceSelector;

class FWHGCalMultiClusterLegoProxyBuilder : public FWCaloDataHistProxyBuilder {
public:
  FWHGCalMultiClusterLegoProxyBuilder();
  ~FWHGCalMultiClusterLegoProxyBuilder() override;

  REGISTER_PROXYBUILDER_METHODS();

  FWHGCalMultiClusterLegoProxyBuilder(const FWHGCalMultiClusterLegoProxyBuilder &) = delete;  // stop default
  const FWHGCalMultiClusterLegoProxyBuilder &operator=(const FWHGCalMultiClusterLegoProxyBuilder &) =
      delete;  // stop default

private:
  void fillCaloData() override;
  FWHistSliceSelector *instantiateSliceSelector() override;
  void build(const FWEventItem *iItem, TEveElementList *product, const FWViewContext *) override;

  const std::vector<reco::HGCalMultiCluster> *m_towers;
};

#endif
