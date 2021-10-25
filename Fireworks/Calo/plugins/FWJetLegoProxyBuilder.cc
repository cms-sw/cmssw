#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

class FWJetLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet> {
public:
  FWJetLegoProxyBuilder() {}
  ~FWJetLegoProxyBuilder() override {}

  REGISTER_PROXYBUILDER_METHODS();

protected:
  using FWSimpleProxyBuilderTemplate<reco::Jet>::build;
  void build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) override;

public:
  FWJetLegoProxyBuilder(const FWJetLegoProxyBuilder&) = delete;                   // stop default
  const FWJetLegoProxyBuilder& operator=(const FWJetLegoProxyBuilder&) = delete;  // stop default
};

void FWJetLegoProxyBuilder::build(const reco::Jet& iData,
                                  unsigned int iIndex,
                                  TEveElement& oItemHolder,
                                  const FWViewContext*) {
  fireworks::addCircle(iData.eta(), iData.phi(), 0.5, 20, &oItemHolder, this);
}

REGISTER_FWPROXYBUILDER(FWJetLegoProxyBuilder, reco::Jet, "Jets", FWViewType::kAllLegoBits | FWViewType::kLegoHFBit);
