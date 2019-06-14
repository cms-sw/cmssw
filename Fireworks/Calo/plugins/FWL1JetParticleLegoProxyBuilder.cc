/*
 *  FWL1JetParticleLegoProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/3/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

class FWL1JetParticleLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<l1extra::L1JetParticle> {
public:
  FWL1JetParticleLegoProxyBuilder() {}
  ~FWL1JetParticleLegoProxyBuilder() override {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  FWL1JetParticleLegoProxyBuilder(const FWL1JetParticleLegoProxyBuilder&) = delete;                   // stop default
  const FWL1JetParticleLegoProxyBuilder& operator=(const FWL1JetParticleLegoProxyBuilder&) = delete;  // stop default

  using FWSimpleProxyBuilderTemplate<l1extra::L1JetParticle>::build;
  void build(const l1extra::L1JetParticle& iData,
             unsigned int iIndex,
             TEveElement& oItemHolder,
             const FWViewContext*) override;
};

void FWL1JetParticleLegoProxyBuilder::build(const l1extra::L1JetParticle& iData,
                                            unsigned int iIndex,
                                            TEveElement& oItemHolder,
                                            const FWViewContext*) {
  fireworks::addCircle(iData.eta(), iData.phi(), 0.5, 6, &oItemHolder, this);
}

REGISTER_FWPROXYBUILDER(FWL1JetParticleLegoProxyBuilder,
                        l1extra::L1JetParticle,
                        "L1JetParticle",
                        FWViewType::kAllLegoBits);
